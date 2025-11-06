from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime

class VCurrentListsModel(BaseModel):
    id: Optional[int] = None
    name: Optional[str] = None
    details: Optional[str] = None