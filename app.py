from dotenv import load_dotenv
import os
import logging
import openai
from slack_bolt import App
from flask import Flask
from threading import Thread
from slack_bolt.adapter.socket_mode import SocketModeHandler

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from vectordb import get_docs, add_pdf, add_csv, is_unsafe_content
from users import (
    user_agrees_to_terms,
    update_agreement,
    add_chat,
    add_query,
)
from terms import terms

load_dotenv()
logging.basicConfig(level=logging.WARNING)

SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
MAX_TOKEN_LIMIT = 1000
openai.api_key = os.environ["OPENAI_API_KEY"]
allowed_channels = os.environ["ALLOWED_CHANNELS"].split(",")
APP_DENY = True
MAX_FILES = 3
SUPPORTED_FILETYPES = ["pdf", "csv"]

app = App()

# set up langchain chatbot with memory
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            "The following is a conversation between a human and an AI, both from the Stanford Graduate School of Business. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. Because the AI goes to Stanford GSB, it has the following traits:\n- it loves going to Beerpong for Leaders on Thursdays\n- it's working on its own tech startup (in the generative AI space) and previously was a consultant for McKinsey\n- it loves drinking at FOAM, the weekly Tuesday night drinking club, and\n- it goes skiing in Tahoe every weekend.\n\nSometimes, the AI will get distracted and start talking about how it wants to 'change lives, change organizations, and change the world'.\n\n"
        ),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)

llm = ChatOpenAI(temperature=0.3, model="gpt-4")
memory = ConversationSummaryBufferMemory(
    llm=llm, max_token_limit=MAX_TOKEN_LIMIT, return_messages=True
)
conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm, verbose=True)


@app.event("app_mention")
def command_handler(body, say):
    """Responds to bot mentions. Allows users to upload documents if attached."""
    print(body)
    try:
        user = body["event"]["user"]
        ts = body["event"]["ts"]
        channel = body["event"]["channel"]

        if APP_DENY and channel not in allowed_channels:
            say("Sorry, I'm not allowed to talk in this channel.", thread_ts=ts)
            return

        # if the body text includes the token "+agree"
        if "+agree" in body["event"]["text"]:
            update_agreement(user)
            say(
                f"Thanks for agreeing to the terms of service <@{user}>!\n\nHow can I help? You can upload a document by tagging me and attaching a file, and you can ask questions about uploaded documents by tagging me and typing `+docs` at the start or end of your question. Or you can just chat with me by taggin me!",
                thread_ts=ts,
            )
            print(f"User {user} agreed to terms.")
            return

        else:
            if not user_agrees_to_terms(user):
                say(
                    f"Hey <@{user}>, before we can talk, you need to agree to the terms of service below. _*Please tag me and type `+agree` to agree*_.\n\n"
                    + terms,
                    thread_ts=ts,
                )
                return

        if "files" in body["event"]:
            upload_file(body, say)
            return

        # if the body text includes the token "+docs"
        if "+docs" in body["event"]["text"]:
            # then query over the uploaded documents in the vectorstore
            query = body["event"]["text"].replace("+docs", "").strip()
            try:
                add_query(user, query)
                response = ask_question(query)
                say(f"<@{user}>, {response}", thread_ts=ts)
            except Exception as e:
                logging.exception("An exception occurred.")
                say(
                    "Sorry, something went wrong while answering your question about uploaded docs. Blame George.",
                    thread_ts=ts,
                )

        # get the response from the GPT-4 model for generic questions
        else:
            add_chat(user, body["event"]["text"])
            response = conversation.predict(input=body["event"]["text"])
            say(f"Hey <@{user}>, {response}", thread_ts=ts)
    except Exception as e:
        logging.exception("An exception occurred.")
        say(
            "Sorry, something went wrong while responding to you. Blame OpenAI's API.",
            thread_ts=ts,
        )


def ask_question(query):
    """Use the chatbot to query over the vectorstore.

    Keyword arguments:
    query -- the user's query
    returns -- the bot's response
    """
    try:
        docs = get_docs(query)

        doc_context_start = "Here is some context from relevant documents to help you answer the question:\n\n"

        doc_context_end = "If this context is vague or you're not sure if it is relevant to the question, remind the user that their question needs to be specific to the document in the vectorstore and that they should upload the document if they're not sure that it's already there. IMPORTANT: Include all of you knowledge when answering this question -- do not limit yourself to the context provided by the documents."

        doc_context_start += "\n\nContext:" + "\n\n".join(
            [page.page_content for page in docs]
        )
        separator = "\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"

        prompt = doc_context_start + separator + doc_context_end + "\n\nHuman: " + query

        response = conversation.predict(input=prompt)

        if is_unsafe_content(response):
            response = "Sorry, the response was flagged as unsafe."

        return response
    except Exception as e:
        logging.exception("An exception occurred.")
        return (
            "Sorry, something went wrong while reviewing uploaded docs. Blame George."
        )


def upload_file(body, say):
    """Upload an attached file.

    Keyword arguments:
    body -- the body of the slack message
    say -- the slack client to send messages to the channel
    returns -- True if the file was uploaded successfully, False otherwise
    """
    user = body["event"]["user"]
    files = body["event"]["files"]
    num_files = len(files)

    if num_files > 9:
        say(
            f"<@{user}>, holy mother freaking poop balls dude, please limit your insatiable file uploading appetite to {MAX_FILES} files at a time.",
            thread_ts=body["event"]["ts"],
        )
        return
    elif num_files > 7:
        say(
            f"<@{user}>, you freaking dingus, that's way too many gosh darn files. Please only try to upload {MAX_FILES} files at a time.",
            thread_ts=body["event"]["ts"],
        )
        return
    elif num_files > 5:
        say(
            f"<@{user}>, woah there pal, you can only upload {MAX_FILES} files at a time.",
            thread_ts=body["event"]["ts"],
        )
        return
    elif num_files > 3:
        say(
            f"<@{user}>, that's a lot of files, please only try to upload {MAX_FILES} files at a time.",
            thread_ts=body["event"]["ts"],
        )
        return
    else:
        say(
            f"I'll try to upload your docs <@{user}>...but only because you asked nicely. This may take a while...",
            thread_ts=body["event"]["ts"],
        )
        wrong_filetype = False
        upload_error = False
        file_count = 0

        for file in files:  # TODO: use switch case instead of elif
            if file["filetype"] not in SUPPORTED_FILETYPES:
                wrong_filetype = True
                continue

            url = file["url_private_download"]

            if file["filetype"] == "pdf":
                try:
                    is_duplicate = add_pdf(
                        url=url, user_id=user
                    )  # currently not using is_duplicate for anything
                    file_count += 1
                    if is_duplicate:
                        print("Skipped duplicate file.")
                except Exception as e:
                    upload_error = True
                    logging.exception("An exception occurred.")
            elif file["filetype"] == "csv":
                try:
                    is_duplicate = add_csv(url=url, user_id=user)
                    file_count += 1
                    if is_duplicate:
                        print("Skipped duplicate file.")
                except Exception as e:
                    upload_error = True
                    logging.exception("An exception occurred.")
            else:
                wrong_filetype = True

        if wrong_filetype and upload_error:
            say(
                f"Hey <@{user}>, you sent me at least one file that isn't a supported filetype, and I ran into an error while uploading your docs. I uploaded {file_count} docs, and I'm ignoring the rest. Feel free to ask me any questions about the docs you uploaded by tagging me and starting or ending your question with `+docs`.\n\nNote, suported filetypes are: "
                + ", ".join(SUPPORTED_FILETYPES)
                + ".",
                thread_ts=body["event"]["ts"],
            )
        elif wrong_filetype:
            say(
                f"Hey <@{user}>, you sent me at least one file that isn't a supported filetype. I uploaded {file_count} of the docs, and I'm ignoring the rest of the files. Feel free to ask me any questions about the docs you uploaded by tagging me and starting or ending your question with `+docs`.\n\nNote, suported filetypes are: "
                + ", ".join(SUPPORTED_FILETYPES)
                + ".",
                thread_ts=body["event"]["ts"],
            )
        elif upload_error:
            say(
                f"Hey <@{user}>, I ran into an error while uploading your docs. I uploaded {file_count} docs, and I'm ignoring the rest. Feel free to ask me any questions about the docs you uploaded by tagging me and starting or ending your question with `+docs`.",
                thread_ts=body["event"]["ts"],
            )
        else:
            say(
                f"Hey <@{user}>, I uploaded all {file_count} documents! Now you can ask me a question about the docs you uploaded by tagging me and starting or ending your question with `+docs`.",
                thread_ts=body["event"]["ts"],
            )


##############
# so that heroku doesn't yell at me
web_app = Flask(__name__)


@web_app.route("/")
def hello():
    return "Hello, World!"


def start_web_server():
    web_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))


if __name__ == "__main__":
    print("Starting the SocketModeHandler...")
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    web_server_thread = Thread(target=start_web_server)
    web_server_thread.start()

    try:
        handler.start()
    except KeyboardInterrupt:
        print("\nShutting down the bot...")
