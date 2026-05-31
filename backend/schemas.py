from pydantic import BaseModel

class ImagePayload(BaseModel):
    image: str
