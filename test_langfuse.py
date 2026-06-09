from dotenv import load_dotenv

load_dotenv()

from langfuse import Langfuse

langfuse = Langfuse()

trace = langfuse.trace(name="test", input="hello", output="world")

langfuse.flush()
print("Done — check Langfuse dashboard")
