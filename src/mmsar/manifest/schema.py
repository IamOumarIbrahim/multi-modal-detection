"""Manifest schema dataclasses for tracking dataset collection and annotation progress."""

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Bin:
    """Represents a single condition x scenario dataset partition bin."""

    condition: str
    scenario: str
    count: int = 0
    target: int = 5

    @property
    def is_complete(self) -> bool:
        return self.count >= self.target

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "scenario": self.scenario,
            "count": self.count,
            "target": self.target,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Bin":
        return cls(
            condition=data["condition"],
            scenario=data["scenario"],
            count=int(data.get("count", 0)),
            target=int(data.get("target", 5)),
        )


@dataclass
class Manifest:
    """Represents dataset manifest across all 9 bins (3 conditions x 3 scenarios)."""

    bins: dict[str, dict[str, Bin]] = field(default_factory=dict)

    CONDITIONS: list[str] = field(
        default_factory=lambda: ["desert", "forest", "altitude"], init=False
    )
    SCENARIOS: list[str] = field(
        default_factory=lambda: ["positive", "hard_negative", "clear_negative"],
        init=False,
    )

    @classmethod
    def create_empty(cls) -> "Manifest":
        """Create an empty manifest with target=5 for all 9 bins."""
        manifest = cls()
        for cond in ["desert", "forest", "altitude"]:
            manifest.bins[cond] = {}
            for scen in ["positive", "hard_negative", "clear_negative"]:
                manifest.bins[cond][scen] = Bin(
                    condition=cond,
                    scenario=scen,
                    count=0,
                    target=5,
                )
        return manifest

    def get_bin(self, condition: str, scenario: str) -> Bin:
        return self.bins[condition][scenario]

    @property
    def total_count(self) -> int:
        return sum(b.count for cond in self.bins.values() for b in cond.values())

    @property
    def total_target(self) -> int:
        return sum(b.target for cond in self.bins.values() for b in cond.values())

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for cond, scenarios in self.bins.items():
            result[cond] = {}
            for scen, b in scenarios.items():
                result[cond][scen] = b.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        manifest = cls()
        for cond, scenarios in data.items():
            manifest.bins[cond] = {}
            for scen, b_data in scenarios.items():
                manifest.bins[cond][scen] = Bin.from_dict(b_data)
        return manifest
