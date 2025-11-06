from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime

class EventInvitationEntriesModel(BaseModel):
    entry_id: Optional[int] = None
    version_id: int
    event_name: str
    event_date: date
    invitee_id: str
    invitee_name: Optional[str] = None
    email: str
    status: Optional[str] = None