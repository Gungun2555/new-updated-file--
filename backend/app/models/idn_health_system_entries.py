from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime

class IdnHealthSystemEntriesModel(BaseModel):
    entry_id: Optional[int] = None
    version_id: int
    system_id: str
    system_name: str
    contact_name: Optional[str] = None
    contact_email: Optional[str] = None
    importance: Optional[str] = None