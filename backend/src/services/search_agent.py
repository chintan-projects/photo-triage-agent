"""LFM-powered conversational search agent for photo queries.

Follows CLAUDE.md principles:
- Robustness: Timeout handling, graceful fallbacks
- Observability: Structured logging with context
- Reusability: Cached context, shared utilities
"""
import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field

import aiosqlite
import structlog

from ..classifiers import LFMCliProvider
from ..config import get_model_config, models_available
from ..database import AnalysisRepository, PhotoRepository

logger = structlog.get_logger()

# Default timeout for LLM reasoning (seconds)
DEFAULT_LLM_TIMEOUT = 30.0

# Maximum photos to include in LLM context
MAX_CONTEXT_PHOTOS = 100

# =============================================================================
# Prompts - Tighter, more focused for better results
# =============================================================================

SEARCH_PROMPT = """You are analyzing a photo library to find matching photos.

LIBRARY SUMMARY:
- Total photos: {total_photos}
- Categories: {categories}

CANDIDATE PHOTOS (pre-filtered by keywords):
{photo_context}

USER QUERY: "{query}"
{conversation_context}

INSTRUCTIONS:
1. Understand what the user is looking for
2. Match against the candidate photo descriptions
3. Return ONLY a JSON object (no explanation):

{{
  "matches": [
    {{"id": <photo_id>, "score": 0.0-1.0, "reason": "brief reason"}}
  ],
  "interpretation": "what you understood the user wants",
  "confidence": 0.0-1.0,
  "clarifying_question": null or "question if query is ambiguous",
  "suggestions": ["optional follow-up queries"]
}}

Be selective - only include photos with score > 0.5.
Limit matches to the most relevant 20 photos."""

REFINEMENT_PROMPT = """You are refining a photo search based on user feedback.

PREVIOUS RESULTS:
- Found {result_count} photos matching: "{previous_interpretation}"
- Previous matches: {previous_ids}

USER FEEDBACK: "{feedback}"

CANDIDATE PHOTOS:
{photo_context}

Return ONLY a JSON object:
{{
  "matches": [
    {{"id": <photo_id>, "score": 0.0-1.0, "reason": "brief reason"}}
  ],
  "interpretation": "updated understanding",
  "confidence": 0.0-1.0,
  "action": "narrow" | "expand" | "restructure"
}}

Apply the user's feedback to filter or expand the previous results."""

EXPLAIN_PROMPT = """Analyze these photo descriptions and explain what they have in common:

{descriptions}

Provide a brief (1-2 sentence) explanation of what connects these photos.
Be specific about visual elements, themes, or subjects they share."""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SearchResult:
    """Result from a search query."""
    interpretation: str
    photos: list
    total_found: int
    confidence: float
    clarifying_question: str | None = None
    search_criteria: dict | None = None
    suggestions: list[str] | None = None


@dataclass
class PhotoContext:
    """Cached photo library context for LLM reasoning."""
    photo_count: int
    categories: dict[str, int]
    descriptions: dict[int, str]  # photo_id -> description
    keywords_index: dict[str, set[int]]  # keyword -> photo_ids
    cache_hash: str
    created_at: float = field(default_factory=time.time)

    def is_stale(self, max_age_seconds: float = 300) -> bool:
        """Check if cache is older than max_age_seconds (default 5 min)."""
        return (time.time() - self.created_at) > max_age_seconds


# =============================================================================
# PhotoSearchAgent
# =============================================================================

class PhotoSearchAgent:
    """Uses LFM to reason about photo queries and return matching results.

    Features:
    - Caches photo context to avoid rebuilding on every query
    - Pre-filters with keywords before sending to LLM (reduces context size)
    - Timeout handling with graceful fallback to keyword search
    - Structured logging for observability
    """

    def __init__(
        self,
        db: aiosqlite.Connection,
        model_provider: LFMCliProvider | None = None,
        llm_timeout: float = DEFAULT_LLM_TIMEOUT,
    ):
        self.db = db
        self.photo_repo = PhotoRepository(db)
        self.analysis_repo = AnalysisRepository(db)
        self._model = model_provider
        self._llm_timeout = llm_timeout
        self._context_cache: PhotoContext | None = None

    async def _ensure_model(self) -> LFMCliProvider | None:
        """Lazy-load model provider.

        Returns:
            LFMCliProvider if models available, None otherwise.
        """
        if self._model is None:
            if not models_available():
                logger.warning("models_not_available")
                return None
            config = get_model_config()
            self._model = LFMCliProvider(
                model_path=config["model_path"],
                mmproj_path=config["mmproj_path"],
                cli_path=config["cli_path"],
                n_ctx=config["n_ctx"],
                n_gpu_layers=config["n_gpu_layers"],
            )
        return self._model

    # =========================================================================
    # Context Caching (Critical for Performance)
    # =========================================================================

    async def _compute_library_hash(self) -> str:
        """Compute hash of library state for cache invalidation."""
        count = await self.photo_repo.count()
        categories = await self.photo_repo.get_category_counts()
        state = f"{count}:{sorted(categories.items())}"
        return hashlib.md5(state.encode()).hexdigest()

    async def _build_photo_context(self) -> PhotoContext:
        """Build full photo context with descriptions and keyword index.

        This is expensive - only called when cache is stale or missing.
        """
        start = time.time()
        logger.info("building_photo_context")

        # Get all photos with their analysis results
        photos = await self.photo_repo.get_all(limit=10000)
        categories = await self.photo_repo.get_category_counts()

        descriptions: dict[int, str] = {}
        keywords_index: dict[str, set[int]] = {}

        for photo in photos:
            # Always index by filename (fallback when no LFM descriptions)
            filename_words = set(photo.filename.lower().replace("_", " ").replace("-", " ").split())
            for word in filename_words:
                if len(word) > 2:
                    if word not in keywords_index:
                        keywords_index[word] = set()
                    keywords_index[word].add(photo.id)

            # Try to get LFM analysis for richer descriptions
            analyses = await self.analysis_repo.get_for_photo(photo.id)
            for analysis in analyses:
                if analysis.analyzer == "lfm":
                    desc = analysis.result.get("description", "")
                    category = analysis.result.get("category", "")
                    if desc:
                        descriptions[photo.id] = desc
                        # Index keywords for fast pre-filtering
                        words = set(desc.lower().split())
                        if category:
                            words.add(category.lower())
                        for word in words:
                            if len(word) > 2:  # Skip tiny words
                                if word not in keywords_index:
                                    keywords_index[word] = set()
                                keywords_index[word].add(photo.id)
                    elif category:
                        # Even without description, use category
                        descriptions[photo.id] = f"Photo categorized as: {category}"
                        if category.lower() not in keywords_index:
                            keywords_index[category.lower()] = set()
                        keywords_index[category.lower()].add(photo.id)
                    break

        cache_hash = await self._compute_library_hash()

        elapsed = time.time() - start
        logger.info(
            "photo_context_built",
            photo_count=len(photos),
            descriptions=len(descriptions),
            keywords=len(keywords_index),
            elapsed_ms=round(elapsed * 1000),
        )

        # Warn if no descriptions - search will be limited
        if len(photos) > 0 and len(descriptions) == 0:
            logger.warning(
                "no_lfm_descriptions",
                message="No LFM descriptions found. Re-run analysis with skip_lfm=false for better search.",
            )

        return PhotoContext(
            photo_count=len(photos),
            categories=categories,
            descriptions=descriptions,
            keywords_index=keywords_index,
            cache_hash=cache_hash,
        )

    async def _get_context(self) -> PhotoContext:
        """Get cached photo context, rebuilding if stale."""
        current_hash = await self._compute_library_hash()

        if (
            self._context_cache is None
            or self._context_cache.is_stale()
            or self._context_cache.cache_hash != current_hash
        ):
            self._context_cache = await self._build_photo_context()

        return self._context_cache

    def invalidate_cache(self) -> None:
        """Force cache invalidation (call after photos are added/removed)."""
        self._context_cache = None
        logger.debug("context_cache_invalidated")

    # =========================================================================
    # Keyword Pre-Filtering (Reduces LLM Context Size)
    # =========================================================================

    async def _keyword_prefilter(
        self,
        query: str,
        limit: int = MAX_CONTEXT_PHOTOS,
    ) -> list[int]:
        """Fast keyword-based pre-filtering before LLM reasoning.

        This dramatically reduces the context sent to the LLM.
        """
        context = await self._get_context()
        query_words = set(query.lower().split())

        # Score each photo by keyword matches
        scores: dict[int, int] = {}
        for word in query_words:
            if len(word) <= 2:
                continue
            # Exact match
            if word in context.keywords_index:
                for photo_id in context.keywords_index[word]:
                    scores[photo_id] = scores.get(photo_id, 0) + 2
            # Partial match (prefix)
            for indexed_word, photo_ids in context.keywords_index.items():
                if indexed_word.startswith(word) or word.startswith(indexed_word):
                    for photo_id in photo_ids:
                        scores[photo_id] = scores.get(photo_id, 0) + 1

        # Sort by score and return top candidates
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return sorted_ids[:limit]

    def _format_photo_context(
        self,
        photo_ids: list[int],
        context: PhotoContext,
    ) -> str:
        """Format photo descriptions for LLM prompt."""
        if not photo_ids:
            return "(No matching photos found)"

        lines = []
        for pid in photo_ids[:MAX_CONTEXT_PHOTOS]:
            desc = context.descriptions.get(pid, "No description")
            lines.append(f"- Photo {pid}: {desc}")

        return "\n".join(lines)

    # =========================================================================
    # JSON Extraction
    # =========================================================================

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from model response with fallback."""
        # Try to find JSON in the response
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: empty result
        logger.warning("json_extraction_failed", response_preview=text[:100])
        return {
            "matches": [],
            "interpretation": "Could not parse search criteria",
            "confidence": 0.3,
            "clarifying_question": "Could you rephrase your search query?",
        }

    # =========================================================================
    # LLM Search with Timeout
    # =========================================================================

    async def _llm_search(
        self,
        query: str,
        candidate_ids: list[int],
        context: PhotoContext,
        conversation_context: str = "",
    ) -> dict:
        """Run LLM reasoning on candidate photos with timeout.

        Raises:
            asyncio.TimeoutError: If LLM takes too long
            RuntimeError: If model not available
        """
        model = await self._ensure_model()
        if model is None:
            raise RuntimeError("Model not available")

        photo_context = self._format_photo_context(candidate_ids, context)

        prompt = SEARCH_PROMPT.format(
            total_photos=context.photo_count,
            categories=", ".join(f"{k}({v})" for k, v in context.categories.items()) or "none",
            photo_context=photo_context,
            query=query,
            conversation_context=conversation_context,
        )

        # Run with timeout
        start = time.time()

        def _sync_reason():
            return model.reason(prompt)

        loop = asyncio.get_event_loop()
        response = await asyncio.wait_for(
            loop.run_in_executor(None, _sync_reason),
            timeout=self._llm_timeout,
        )

        elapsed = time.time() - start
        logger.info("llm_search_completed", elapsed_ms=round(elapsed * 1000))

        return self._extract_json(response)

    async def _keyword_fallback(
        self,
        query: str,
        limit: int = 50,
    ) -> SearchResult:
        """Fallback to pure keyword search when LLM fails/times out."""
        logger.info("using_keyword_fallback", query=query)

        candidate_ids = await self._keyword_prefilter(query, limit=limit)

        # If still no matches, return all photos so user sees something
        if not candidate_ids:
            logger.info("keyword_fallback_all_photos", reason="no matches found")
            all_photos = await self.photo_repo.get_all(limit=limit)
            return SearchResult(
                interpretation=f"Showing all photos (no matches found for: {query})",
                photos=all_photos,
                total_found=len(all_photos),
                confidence=0.1,
                search_criteria={"keywords": query.lower().split()},
                suggestions=["Try different keywords", "Make sure photos are analyzed with LFM"],
            )

        photos = []
        for pid in candidate_ids:
            photo = await self.photo_repo.get_by_id(pid)
            if photo:
                photos.append(photo)

        return SearchResult(
            interpretation=f"Keyword search for: {query}",
            photos=photos[:limit],
            total_found=len(photos),
            confidence=0.3,  # Lower confidence for keyword-only
            search_criteria={"keywords": query.lower().split()},
        )

    # =========================================================================
    # Public API
    # =========================================================================

    async def search(
        self,
        query: str,
        conversation_history: list[dict] | None = None,
        limit: int = 50,
    ) -> SearchResult:
        """Search photos using natural language query.

        Flow:
        1. Keyword pre-filter to get candidates (fast)
        2. LLM reasoning on candidates (with timeout)
        3. Fallback to keyword search if LLM fails

        Args:
            query: Natural language search query.
            conversation_history: Previous messages for context.
            limit: Maximum results to return.

        Returns:
            SearchResult with matching photos and interpretation.
        """
        logger.info("search_query", query=query)
        start = time.time()

        # Get cached context
        context = await self._get_context()

        # Log diagnostic info
        logger.info(
            "search_context",
            photo_count=context.photo_count,
            descriptions_count=len(context.descriptions),
            keywords_count=len(context.keywords_index),
        )

        # Step 1: Keyword pre-filter (fast)
        candidate_ids = await self._keyword_prefilter(query, limit=MAX_CONTEXT_PHOTOS)
        logger.info("prefilter_complete", candidates=len(candidate_ids), query=query)

        # If no keyword matches, fall back to recent photos (better than nothing)
        if not candidate_ids and context.photo_count > 0:
            logger.info("prefilter_fallback", reason="no keyword matches, using all photos")
            # Get all photo IDs from descriptions or just use first N photos
            if context.descriptions:
                candidate_ids = list(context.descriptions.keys())[:MAX_CONTEXT_PHOTOS]
            else:
                # No descriptions at all - get photos directly
                all_photos = await self.photo_repo.get_all(limit=MAX_CONTEXT_PHOTOS)
                candidate_ids = [p.id for p in all_photos]

        # Build conversation context
        conversation_context = ""
        if conversation_history:
            recent = conversation_history[-3:]
            conversation_context = "\nPrevious conversation:\n" + "\n".join(
                f"- {msg['role']}: {msg['content'][:100]}..."
                for msg in recent
            )

        # Step 2: Try LLM reasoning with timeout
        try:
            result = await self._llm_search(
                query=query,
                candidate_ids=candidate_ids,
                context=context,
                conversation_context=conversation_context,
            )

            # Parse matches from LLM response
            matches = result.get("matches", [])
            matched_ids = [m["id"] for m in matches if m.get("score", 0) > 0.5]
            logger.info("llm_matches", count=len(matched_ids), total_in_response=len(matches))

            # Fetch actual photo objects
            photos = []
            for pid in matched_ids[:limit]:
                photo = await self.photo_repo.get_by_id(pid)
                if photo:
                    photos.append(photo)

            # If LLM returned no matches, fall back to keyword candidates
            if not photos and candidate_ids:
                logger.info("llm_no_matches_fallback", using="candidate_ids", count=len(candidate_ids))
                for pid in candidate_ids[:limit]:
                    photo = await self.photo_repo.get_by_id(pid)
                    if photo:
                        photos.append(photo)
                # Adjust interpretation to reflect fallback
                result["interpretation"] = result.get("interpretation", "") or f"Showing photos matching: {query}"
                result["confidence"] = 0.4  # Lower confidence for fallback

            elapsed = time.time() - start
            logger.info(
                "search_complete",
                query=query,
                results=len(photos),
                elapsed_ms=round(elapsed * 1000),
            )

            return SearchResult(
                interpretation=result.get("interpretation", ""),
                photos=photos,
                total_found=len(photos),
                confidence=result.get("confidence", 0.5),
                clarifying_question=result.get("clarifying_question"),
                search_criteria={"matches": matches},
                suggestions=result.get("suggestions"),
            )

        except asyncio.TimeoutError:
            logger.warning("llm_search_timeout", query=query, timeout=self._llm_timeout)
            return await self._keyword_fallback(query, limit)

        except Exception as e:
            logger.error("llm_search_error", error=str(e), query=query)
            return await self._keyword_fallback(query, limit)

    async def refine(
        self,
        feedback: str,
        previous_result: SearchResult,
        limit: int = 50,
    ) -> SearchResult:
        """Refine previous search based on user feedback.

        Args:
            feedback: User's feedback on previous results.
            previous_result: The previous search result to refine.
            limit: Maximum results to return.

        Returns:
            Updated SearchResult with refined matches.
        """
        logger.info("search_refine", feedback=feedback)

        context = await self._get_context()

        # Get previous match IDs
        previous_ids = [p.id for p in previous_result.photos[:20]]

        # Pre-filter with feedback keywords
        candidate_ids = await self._keyword_prefilter(feedback, limit=MAX_CONTEXT_PHOTOS)
        # Also include previous results
        all_candidates = list(set(candidate_ids + previous_ids))

        photo_context = self._format_photo_context(all_candidates, context)

        prompt = REFINEMENT_PROMPT.format(
            result_count=previous_result.total_found,
            previous_interpretation=previous_result.interpretation,
            previous_ids=previous_ids[:10],
            feedback=feedback,
            photo_context=photo_context,
        )

        try:
            model = await self._ensure_model()
            if model is None:
                raise RuntimeError("Model not available")

            def _sync_reason():
                return model.reason(prompt)

            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, _sync_reason),
                timeout=self._llm_timeout,
            )
            result = self._extract_json(response)

            # Parse matches
            matches = result.get("matches", [])
            matched_ids = [m["id"] for m in matches if m.get("score", 0) > 0.5]

            photos = []
            for pid in matched_ids[:limit]:
                photo = await self.photo_repo.get_by_id(pid)
                if photo:
                    photos.append(photo)

            return SearchResult(
                interpretation=result.get("interpretation", ""),
                photos=photos,
                total_found=len(photos),
                confidence=result.get("confidence", 0.5),
                search_criteria={"matches": matches},
            )

        except (asyncio.TimeoutError, Exception) as e:
            logger.error("refine_error", error=str(e))
            # Fallback: filter previous results by feedback keywords
            feedback_words = set(feedback.lower().split())
            filtered = []
            for photo in previous_result.photos:
                desc = context.descriptions.get(photo.id, "").lower()
                if any(w in desc for w in feedback_words):
                    filtered.append(photo)

            return SearchResult(
                interpretation=f"Refined search: {feedback}",
                photos=filtered[:limit],
                total_found=len(filtered),
                confidence=0.3,
            )

    async def explain_group(self, photo_ids: list[int]) -> str:
        """Use LFM to explain why photos are grouped together.

        Args:
            photo_ids: List of photo IDs to analyze.

        Returns:
            Natural language explanation of the group.
        """
        if not photo_ids:
            return "No photos to analyze."

        context = await self._get_context()

        # Get descriptions for photos
        descriptions = []
        for pid in photo_ids[:5]:  # Limit for prompt size
            desc = context.descriptions.get(pid)
            if desc:
                descriptions.append(f"- {desc}")

        if not descriptions:
            return "These photos have not been analyzed yet."

        prompt = EXPLAIN_PROMPT.format(descriptions="\n".join(descriptions))

        try:
            model = await self._ensure_model()
            if model is None:
                raise RuntimeError("Model not available")

            def _sync_reason():
                return model.reason(prompt, max_tokens=256)

            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, _sync_reason),
                timeout=self._llm_timeout,
            )
            return response.strip()

        except Exception as e:
            logger.error("explain_group_error", error=str(e))
            return f"Group of {len(photo_ids)} similar photos."
