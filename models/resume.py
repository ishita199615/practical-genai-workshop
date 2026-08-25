"""Pydantic models for the fictional master resume."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResumeBullet(BaseModel):
    """A single stable-ID bullet inside an experience or project entry."""

    id: str
    text: str


class ExperienceEntry(BaseModel):
    """One experience entry with stable IDs for every bullet."""

    id: str
    title: str
    organization: str
    dates: str | None = None
    bullets: list[ResumeBullet] = Field(default_factory=list)


class ResumeProfile(BaseModel):
    """The complete fictional applicant profile used by the demo."""

    candidate_id: str
    name: str
    email: str
    phone: str | None = None
    location: str
    professional_summary: str
    target_roles: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    education: list[dict] = Field(default_factory=list)
    experience: list[ExperienceEntry] = Field(default_factory=list)
    projects: list[dict] = Field(default_factory=list)
    document_features: dict = Field(default_factory=dict)

    def bullet_index(self) -> dict[str, str]:
        """Return a mapping of every bullet ID to its original text.

        Includes experience bullets and project bullets so that validation can
        confirm a revised bullet points at a real source.
        """
        index: dict[str, str] = {}
        for entry in self.experience:
            for bullet in entry.bullets:
                index[bullet.id] = bullet.text
        for project in self.projects:
            for bullet in project.get("bullets", []) or []:
                bullet_id = bullet.get("id")
                if bullet_id:
                    index[bullet_id] = bullet.get("text", "")
        return index

    def all_bullet_texts(self) -> list[str]:
        """Return every bullet text in the resume."""
        return list(self.bullet_index().values())

    def as_plain_text(self) -> str:
        """Render a canonical, ATS-style plain-text version of the resume.

        This rendering is the single input used by the deterministic ATS
        readiness rubric, so the original and proposed resumes are always
        scored the same way.
        """
        return render_resume_text(
            name=self.name,
            email=self.email,
            phone=self.phone,
            location=self.location,
            summary=self.professional_summary,
            skills=self.skills,
            education=self.education,
            experience=[
                {
                    "title": entry.title,
                    "organization": entry.organization,
                    "dates": entry.dates,
                    "bullets": [bullet.text for bullet in entry.bullets],
                }
                for entry in self.experience
            ],
            projects=[
                {
                    "name": project.get("name", ""),
                    "bullets": [
                        bullet.get("text", "")
                        for bullet in project.get("bullets", []) or []
                    ],
                }
                for project in self.projects
            ],
        )


def render_resume_text(
    *,
    name: str,
    email: str,
    phone: str | None,
    location: str,
    summary: str,
    skills: list[str],
    education: list[dict],
    experience: list[dict],
    projects: list[dict],
) -> str:
    """Render a canonical plain-text resume from raw parts.

    Used for both the original resume and the proposed (patched) resume so the
    ATS rubric compares like with like.
    """
    lines: list[str] = []
    lines.append(name)
    contact = [value for value in (email, phone, location) if value]
    lines.append(" | ".join(contact))
    lines.append("")
    lines.append("PROFESSIONAL SUMMARY")
    lines.append(summary)
    lines.append("")
    lines.append("SKILLS")
    lines.append(", ".join(skills))
    lines.append("")
    lines.append("EDUCATION")
    for entry in education:
        parts = [
            str(entry.get("degree", "")),
            str(entry.get("institution", "")),
            str(entry.get("graduation", "")),
        ]
        lines.append(" — ".join(part for part in parts if part))
    lines.append("")
    lines.append("EXPERIENCE")
    for entry in experience:
        header = [entry.get("title", ""), entry.get("organization", "")]
        if entry.get("dates"):
            header.append(str(entry["dates"]))
        lines.append(" — ".join(part for part in header if part))
        for bullet in entry.get("bullets", []):
            lines.append(f"- {bullet}")
    if projects:
        lines.append("")
        lines.append("PROJECTS")
        for project in projects:
            if project.get("name"):
                lines.append(str(project["name"]))
            for bullet in project.get("bullets", []):
                lines.append(f"- {bullet}")
    return "\n".join(lines).strip()
