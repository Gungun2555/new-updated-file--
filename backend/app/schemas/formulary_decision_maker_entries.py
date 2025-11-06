from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime

class FormularyDecisionMakerEntries(BaseModel):
    entry_id: Optional[int] = None
    version_id: int
    contact_id: str
    contact_name: Optional[str] = None
    organization: str
    email: Optional[str] = None
    influence_level: Optional[str] = None

    class Config:
        orm_mode = True