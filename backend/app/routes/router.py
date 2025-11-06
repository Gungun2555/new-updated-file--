
from fastapi import APIRouter
from .crud import routers as _routers
from .lists import router as lists_router
from .chatbot import router as chatbot_router
from .injection import router as injection_router

router = APIRouter()

# Include the custom lists router
router.include_router(lists_router)

# Include the chatbot router
router.include_router(chatbot_router)
router.include_router(injection_router)


# Include all CRUD routers
for r in _routers:
    router.include_router(r)
