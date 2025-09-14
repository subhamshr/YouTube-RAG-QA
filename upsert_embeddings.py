from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone_client import index
from langchain_openai import OpenAIEmbeddings


video_id = "YSMWN8VpY6A"
ytt_api = YouTubeTranscriptApi()
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

#1.Fetch transcript
try:
    transcript_list = ytt_api.fetch(video_id, languages=["en"])
    transcript = " ".join(snippet.text for snippet in transcript_list.snippets)
except TranscriptsDisabled:
    raise Exception("Captions are disabled for this video.")
except NoTranscriptFound:
    raise Exception("No transcript found for this video.")
except VideoUnavailable:
    raise Exception("The video is unavailable.")

#2.Split transcript into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

#3.Generate embeddings
embeddings_list = [embeddings_model.embed_query(chunk.page_content) for chunk in chunks]

#4.Upsert into Pinecone
vectors_to_upsert = [
    (f"id_{i}", emb, {"text": chunk.page_content})
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings_list))
]
index.upsert(vectors=vectors_to_upsert)
print("✅ Transcript chunks upserted to Pinecone.")
