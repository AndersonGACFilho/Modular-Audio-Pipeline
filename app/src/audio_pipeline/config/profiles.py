"""Reusable analysis profiles and routing independent of LLM providers."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel, Field


PROFILE_INSTRUCTIONS = {
    "interview": (
        "Extract candidate, interviewer, organization, role, experience, skills, "
        "fit signals, concerns, process stage and next steps."
    ),
    "career_mentoring": (
        "Extract mentee goals, current situation, guidance, recommendations, "
        "career options, agreed actions and follow-up."
    ),
    "technical_daily": (
        "Extract work updates, completed items, work in progress, blockers, "
        "tickets or pull requests, owners and next actions."
    ),
    "delivery_planning": (
        "Extract delivery scope, backlog items, priorities, dependencies, risks, "
        "milestones, owners and planned work."
    ),
    "operational_requirements": (
        "Extract business rules, affected systems, integrations, operational flows, "
        "requirements, decisions, risks and implementation tasks."
    ),
    "research_advising": (
        "Extract research objectives, advisor guidance, hypotheses, methods, "
        "experiments, readings, research decisions, risks and student tasks."
    ),
    "research_group_meeting": (
        "Extract participants, presentations, project updates, research discussions, "
        "group decisions, announcements, actions and next meeting preparation."
    ),
    "collaboration_alignment": (
        "Extract participants' expertise, collaboration objective, common interests, "
        "opportunities, agreements, open questions and next steps."
    ),
    "generic_meeting": (
        "Extract the meeting context, decisions, discussion points, action items, "
        "risks and next steps."
    ),
}


class ProfileRouting(BaseModel):
    """Classification used to select a reusable formatting profile."""

    profile: str
    confidence: float = Field(ge=0, le=1)
    organization: Optional[str] = None
    project: Optional[str] = None
    reasoning: str


StructuredClassifier = Callable[[str, Type[BaseModel]], Dict[str, Any]]


class ProfileRouter:
    """Selects a profile without coupling profile rules to an LLM provider."""

    _INTERVIEW_SIGNALS = (
        "recruiter",
        "recruitment",
        "hiring",
        "screening",
        "candidate",
        "interview process",
        "open position",
        "open positions",
        "job opening",
        "job openings",
        "compensation",
        "salary",
        "vacancy",
        "vacancies",
        "recrutador",
        "recrutamento",
        "processo seletivo",
        "candidato",
        "vaga",
        "vagas",
        "remuneração",
        "salário",
    )

    @classmethod
    def _interview_evidence(cls, text: str) -> Optional[ProfileRouting]:
        """Return a high-confidence route when recruitment language is explicit.

        Folder names are often organizational labels (for example, ``Mentorias``),
        rather than the actual type of particular recording.  A combination of
        recruitment signals in the spoken content is therefore stronger evidence.
        """
        normalized = (text or "").casefold()
        signals = [signal for signal in cls._INTERVIEW_SIGNALS if signal in normalized]
        if len(signals) < 2:
            return None

        return ProfileRouting(
            profile="interview",
            confidence=0.9,
            reasoning=(
                "Selected from explicit recruitment evidence in the transcription: "
                + ", ".join(signals[:4])
            ),
        )

    def fallback(self, source_path: Optional[str]) -> ProfileRouting:
        path = (source_path or "").lower()
        if "interview" in path or "kokku" in path:
            profile = "interview"
        elif "mentoria" in path:
            profile = "career_mentoring"
        elif "daily" in path:
            profile = "technical_daily"
        elif "delivery-planning" in path:
            profile = "delivery_planning"
        elif "econtrole" in path:
            profile = "operational_requirements"
        elif "kunumi" in path:
            profile = "collaboration_alignment"
        elif "mestrado" in path and "encontros" in path:
            profile = "research_group_meeting"
        elif "mestrado" in path:
            profile = "research_advising"
        else:
            profile = "generic_meeting"
        return ProfileRouting(
            profile=profile,
            confidence=0.65,
            reasoning="Selected from source path fallback",
        )

    def routing_prompt(self, text: str, source_path: Optional[str]) -> str:
        excerpt = text[: min(len(text), 6_000)]
        return (
            "Classify this transcription into exactly one reusable meeting profile: "
            f"{', '.join(PROFILE_INSTRUCTIONS)}. Return the requested JSON only. "
            "Classify from the spoken content first. The source path is a weak "
            "fallback only and must never override transcript evidence. Use "
            "interview for recruitment, candidate evaluation, vacancies, hiring, "
            "screening, compensation, or a selection process. Use career_mentoring "
            "only for coaching or guidance that is not a hiring evaluation. "
            "Use organization and project only when supported by the content. "
            f"Weak source path hint: {source_path or 'unknown'}.\n\n"
            f"Transcription excerpt:\n{excerpt}"
        )

    def route(
        self,
        text: str,
        source_path: Optional[str],
        classifier: Optional[StructuredClassifier] = None,
    ) -> ProfileRouting:
        """Use the classifier when available, otherwise return a deterministic fallback."""
        content_route = self._interview_evidence(text)
        fallback = content_route or self.fallback(source_path)
        if classifier is None:
            return fallback
        try:
            routing = ProfileRouting(**classifier(self.routing_prompt(text, source_path), ProfileRouting))
            if routing.profile not in PROFILE_INSTRUCTIONS:
                return fallback
            if content_route and routing.profile != content_route.profile:
                return content_route
            return routing
        except Exception:
            return fallback
