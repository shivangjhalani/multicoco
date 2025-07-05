import dataclasses
from enum import auto, Enum
from typing import List, Any

class SeparatorStyle(Enum):
    """Different separator styles."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()
    INTERNVL = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str | None = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if seps[i % 2] is not None:
                        ret += role + ": " + message + seps[i % 2]
                    else:
                        ret += role + ": " + message
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.MPT:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n"
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""
            for i, (role, message) in enumerate(self.messages):
                if i == 0:
                    assert message, "first message should be a user message"
                    message = wrap_sys(self.system) + message if self.system else message
                if message:
                    if type(message) is list:
                        message = ' '.join(message)
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += " "
            return ret
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += message + seps[i % 2]
                else:
                    ret += ""
            return ret
        elif self.sep_style == SeparatorStyle.INTERNVL:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is list:
                        message = ' '.join(message)
                    
                    # Ensure sep is not None before concatenation
                    sep = seps[i % 2]
                    if sep is not None:
                        ret += role + message + sep
                    else:
                        ret += role + message
                else:
                    ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

conv_internvl_v1_1 = Conversation(
    system="A helper assistant that can help me with a variety of tasks. ",
    roles=["<human>", "<bot>"],
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.INTERNVL,
    sep=" ",
    sep2="</s>",
)

def get_conv_template(name: str) -> Conversation:
    if name == "internvl_v1.1":
        return conv_internvl_v1_1.copy()
    else: # internvl2.5
        return conv_internvl_v1_1.copy()

if __name__ == "__main__":
    print(get_conv_template("internvl_v1.1").get_prompt()) 