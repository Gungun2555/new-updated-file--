from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime

class DigitalEngagementEntries(BaseModel):
    entry_id: Optional[int] = None
    version_id: int
    contact_id: str
    contact_name: Optional[str] = None
    email: str
    specialty: Optional[str] = None
    opt_in: Optional[bool] = None

    class Config:
        orm_mode = True