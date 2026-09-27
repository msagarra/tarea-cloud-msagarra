from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=18, le=100)

    job: Literal[
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    ]

    marital: Literal[
        "divorced",
        "married",
        "single",
    ]

    education: Literal[
        "primary",
        "secondary",
        "tertiary",
        "unknown",
    ]

    default: Literal["no", "yes"]

    balance: int

    housing: Literal["no", "yes"]

    loan: Literal["no", "yes"]

    contact: Literal[
        "cellular",
        "telephone",
        "unknown",
    ]

    day: int = Field(..., ge=1, le=31)

    month: Literal[
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]

    campaign: int = Field(..., ge=1)

    pdays: int = Field(..., ge=-1)

    previous: int = Field(..., ge=0)

    poutcome: Literal[
        "failure",
        "other",
        "success",
        "unknown",
    ]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(
        protected_namespaces=()
    )

    prediction: str
    probability_yes: float
    prediction_probability: float
    model_version: str
    timestamp_utc: str
