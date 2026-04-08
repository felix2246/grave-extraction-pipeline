from typing import Annotated, Literal, Optional

from pydantic import BaseModel, BeforeValidator, Field

from grave_extraction.utils import fix_text_encoding


class FoundItem(BaseModel):
    name: str = Field(description="Name des gefundenen Items z.B: 'Drahtring'")
    amount: int = Field(description="Anzahl des gefundenen Items")


class Burial(BaseModel):
    geschlecht: Optional[
        Literal["maennlich", "weiblich", "maennlich_unsicher", "weiblich_unsicher"]
    ] = Field(description="Geschlecht der Bestattung", default=None)

    alter: Optional[Literal["infans", "juvenil", "adult", "matur", "senil"]] = Field(
        description="Alter der Bestattung", default=None
    )

    ritus: Optional[str] = Field(
        description="Art der Bestattung z.B. Körperbestattung oder Brandbestattung. Wenn im Text angegeben ist, dass Skelett vergraben ist, ist es meist eine Körperbestattung.",
        default=None,
    )

    orientierung: Optional[str] = Field(
        description='Orientierung der Bestattung (z. B. "SO–NW")',
        default=None,
    )

    lage: Optional[str] = Field(
        description="Lage der Bestattung im Grab, z.B. 'linker Hocker', 'Rückenstrecker'",
        default=None,
    )

    störung: Optional[Literal["ja", "nein", "unsicher"]] = Field(
        description='Wurde die Bestattung gestört? "ja", wenn ein Archäologe nicht mehr gut mit der Bestattung arbeiten kann z.B. bei dislozierten Skelettresten. "unsicher", wenn im Text als unsicher klassifiziert.',
        default=None,
    )


class GraveRecord(BaseModel):
    fundort: Optional[str] = Field(
        description="Fundort oder Ort des Grabes", default=None
    )

    bezirk: Optional[str] = Field(description="Bezirk des Grabes", default=None)

    grab_identifikation: Optional[str] = Field(
        description="Grabnummer oder Kennzeichnung, wie im Text angegeben. z.B: 'Grab 1'",
        default=None,
    )

    bestattungen: list[Burial] = Field(
        description="Liste der Bestattungen im Grab", default=[]
    )

    grabeinbauten: Optional[str] = Field(
        description="Ob das Grab innen mit bestimmten Materialien verkleidet ist. Wenn ja, gib das Material an.",
        default=None,
    )

    grube_form: Optional[
        Annotated[
            Literal[
                "rechteckig",
                "oval",
                "rund",
                "trapezförmig",
                "birnenförmig",
                "ringförmig",
            ],
            BeforeValidator(fix_text_encoding),
        ]
    ] = Field(description="Form der Grabgrube", default=None)

    grube_länge: Optional[float] = Field(
        description="Länge der Grabgrube (in cm)", default=None
    )

    grube_breite: Optional[float] = Field(
        description="Breite der Grabgrube (in cm)", default=None
    )

    grube_tiefe: Optional[float] = Field(
        description="Tiefe der Grabgrube (in cm)", default=None
    )

    beigaben_liste: list[FoundItem] = Field(
        description="Aufführung aller Funde im Grab, z.b. eine Kette oder Goldring",
        default=[],
    )

    referenzierte_abbildungen: list[str] = Field(
        description="Liste aller referenzierten/erwähnten Abbildungen und Tafeln z.B. 'Abb. 2' oder 'Taf. 51 – C:'",
        default=[],
    )


class GraveRecords(BaseModel):
    graves: list[GraveRecord] = []
