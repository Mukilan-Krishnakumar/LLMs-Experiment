import os

from langchain.schema import messages
from apikey import OPENAI_KEY
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


class Character(BaseModel):
    name: str = Field(description="name of the character")
    outfit: str = Field(description="outfit of the character")


os.environ["OPENAI_API_KEY"] = OPENAI_KEY


OPENAI_MODEL = "gpt-3.5-turbo"
PROMPT_CHARACTER_INFO = """
    Provide information about {character} from One Piece
    {format_instructions}
"""


def main():
    llm = ChatOpenAI(model_name=OPENAI_MODEL)
    parser = PydanticOutputParser(pydantic_object=Character)

    character = input("Enter the character name from One Piece: \n")
    message = HumanMessagePromptTemplate.from_template(
        template=PROMPT_CHARACTER_INFO)

    chat_prompt = ChatPromptTemplate.from_messages(messages=[message])
    chat_prompt_filled = chat_prompt.format_prompt(
        character=character, format_instructions=parser.get_format_instructions())

    result = llm(chat_prompt_filled.to_messages())
    data = parser.parse(result.content)
    print(data)


if __name__ == "__main__":
    main()
