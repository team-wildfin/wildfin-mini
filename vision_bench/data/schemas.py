from pydantic import BaseModel

class Behavior(BaseModel):
    '''
    A behavior in the BORIS annotation format.
    '''
    name: str
    category: str
    type: str
    def __hash__(self):
        return hash((self.category, self.name, self.type))

    def __eq__(self, other):
        return isinstance(other, Behavior) and (self.name, self.category, self.type) == (other.name, other.category, other.type)

    
class Event(BaseModel):
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    behavior: Behavior
    subject: str


class Metadata(BaseModel):
    '''
    Metadata for a BORIS annotated video.
    '''
    observation_id: str
    observation_date: str
    # observation_type: str
    # source: str
    fps: float
    media_duration: float
    time_offset: float 


