import os
import logging
import re
import uuid
from typing import List, Dict, Optional
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from mistralai import Mistral
import pinecone
from tqdm.auto import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_environment() -> tuple[str, str]:
    """Initialize environment variables for Google Cloud and return project ID and key path."""
    # key_path = "/content/colab-vertexai-runner.json"
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
    # logger.info(f"Using Service Account key file: {key_path}")
    
    gcp_project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "silicon-works-449817-n7")
    os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_project_id
    logger.info(f"Google Cloud Project set to: {gcp_project_id}")
    
    return gcp_project_id

def initialize_mistral_client(api_key: str) -> Mistral:
    """Initialize and return Mistral client."""
    return Mistral(api_key=api_key)

def process_pdf_ocr(client: Mistral, pdf_path: str) -> str:
    """Process PDF through Mistral OCR and return markdown content."""
    logger.info(f"Uploading PDF: {pdf_path}")
    uploaded_pdf = client.files.upload(
        file={
            "file_name": pdf_path,
            "content": open(pdf_path, "rb"),
        },
        purpose="ocr"
    )

    logger.info(f"Retrieving file ID: {uploaded_pdf.id}")
    retrieved_file = client.files.retrieve(file_id=uploaded_pdf.id)
    
    logger.info("Generating signed URL")
    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
    
    logger.info("Processing OCR")
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        }
    )

    markdown_content = ""
    for page in ocr_response.pages:
        markdown_content += page.markdown + "\n\n"
    
    logger.info("OCR processing completed")
    return markdown_content

def chunk_feedback_by_qa(markdown_content: str, document_metadata: Dict) -> List[Dict]:
    """
    Chunk Markdown feedback text based on Q&A structure, cleaning noise components.
    """
    logger.info("Starting chunking process.")
    chunks = []
    current_question = "General Comment"
    lines = markdown_content.splitlines()
    answer_buffer = []
    current_answer_id = None

    comprehensive_noise_pattern = re.compile(
        r'\\boldsymbol|\\star|\$|\{|\}|[\u2605\u2606\u2730\u24EA\u3147]',
        re.IGNORECASE | re.UNICODE
    )
    question_pattern = re.compile(r"^(?:##? )?Q:\s*(.*)", re.IGNORECASE)
    answer_start_pattern = re.compile(r"^\s*(\d+| \d+[\.\s]|\*|\-)\s*(.*)")
    table_answer_start_pattern = re.compile(r"^\s*\|\s*(\d+)\s*\|\s*(.*)\s*\|")
    header_comment_pattern = re.compile(r"\(\s*\d+\s+comments\s*\)\s*$", re.IGNORECASE)

    def process_buffer():
        nonlocal answer_buffer, current_question, current_answer_id, chunks
        if answer_buffer:
            full_answer_text = "\n".join(answer_buffer).strip()
            if full_answer_text:
                chunk_id = str(uuid.uuid4())
                cleaned_answer_id = str(current_answer_id).strip('. ') if current_answer_id else None
                metadata = {
                    **document_metadata,
                    "question": current_question,
                    "answer_id": cleaned_answer_id,
                }
                chunks.append({
                    "id": chunk_id,
                    "text": full_answer_text,
                    "metadata": metadata
                })
        answer_buffer = []

    for raw_line in lines:
        cleaned_line = comprehensive_noise_pattern.sub('', raw_line)
        line = cleaned_line.strip()

        if not line or line.startswith("![img-") or line.startswith("---") or line == '| :--: | :--: |':
            continue

        if header_comment_pattern.search(line):
            continue

        question_match = question_pattern.match(line)
        if question_match:
            process_buffer()
            current_question = question_match.group(1).strip()
            current_answer_id = None
            continue

        answer_match = answer_start_pattern.match(line)
        table_match = table_answer_start_pattern.match(line)
        match_found = answer_match or table_match

        if match_found:
            process_buffer()
            if answer_match:
                current_answer_id = answer_match.group(1)
                answer_text = answer_match.group(2).strip()
            else:
                current_answer_id = table_match.group(1).strip()
                answer_text = table_match.group(2).strip()

            if answer_text:
                answer_buffer.append(answer_text)
        elif answer_buffer:
            if not (line.startswith("|") and line.strip("|").strip() == ':--:'):
                answer_buffer.append(line)

    process_buffer()
    logger.info(f"Chunking completed. Generated {len(chunks)} chunks.")
    return chunks

def initialize_vertex_ai(project_id: str, location: str = "us-central1") -> TextEmbeddingModel:
    """Initialize Vertex AI and return TextEmbeddingModel."""
    logger.info(f"Initializing Vertex AI for project {project_id} in {location}...")
    aiplatform.init(project=project_id, location=location)
    
    model_name = "text-embedding-004"
    try:
        logger.info(f"Loading TextEmbeddingModel: {model_name}")
        model = TextEmbeddingModel.from_pretrained(model_name)
        logger.info("Successfully loaded embedding model.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}", exc_info=True)
        raise

def add_embeddings_to_chunks(chunks_list: List[Dict], model: TextEmbeddingModel) -> Optional[List[Dict]]:
    """Add embeddings to chunks using Vertex AI TextEmbeddingModel."""
    if not chunks_list:
        logger.warning("Received empty list of chunks. Returning.")
        return chunks_list

    texts_to_embed = [chunk['text'] for chunk in chunks_list]
    logger.info(f"Prepared {len(texts_to_embed)} texts for embedding.")

    batch_size = 250
    all_embeddings = []

    try:
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1} with {len(batch_texts)} texts...")
            
            inputs = [TextEmbeddingInput(text=text, task_type="RETRIEVAL_DOCUMENT") for text in batch_texts]
            embeddings_response = model.get_embeddings(inputs)
            batch_embeddings = [embedding.values for embedding in embeddings_response]
            all_embeddings.extend(batch_embeddings)
            logger.info(f"Received {len(batch_embeddings)} embeddings for batch.")

        if len(all_embeddings) != len(chunks_list):
            logger.error(f"Mismatch: chunks ({len(chunks_list)}) vs embeddings ({len(all_embeddings)}).")
            return None

        for i, chunk in enumerate(chunks_list):
            chunk['embedding'] = all_embeddings[i]
        
        logger.info("Successfully added embeddings to all chunks.")
        return chunks_list
    except Exception as e:
        logger.error(f"Error during embedding generation: {e}", exc_info=True)
        return None

def initialize_pinecone(api_key: str, environment: str, index_name: str) -> Optional[pinecone.Index]:
    """Initialize Pinecone client and connect to index."""
    if not api_key or api_key == "pinecone_api_key":
        logger.error("Pinecone API Key is not set.")
        return None
    
    logger.info("Initializing Pinecone client...")
    try:
        pc = pinecone.Pinecone(api_key=api_key)
        existing_indexes = pc.list_indexes().names()

        if index_name not in existing_indexes:
            logger.error(f"Index '{index_name}' does not exist.")
            return None
        
        logger.info(f"Connecting to existing index: {index_name}")
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        logger.info(f"Index stats: {stats}")
        return index
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {e}", exc_info=True)
        return None

def store_chunks_in_pinecone(chunks_to_store: List[Dict], pinecone_index: pinecone.Index, index_name: str, batch_size: int = 100) -> bool:
    """Store chunks with embeddings in Pinecone index."""
    if not pinecone_index:
        logger.error("Pinecone index object is not available.")
        return False

    if not chunks_to_store:
        logger.warning("Received empty list of chunks to store.")
        return True

    num_chunks = len(chunks_to_store)
    logger.info(f"Preparing to upsert {num_chunks} chunks to Pinecone index '{index_name}'...")

    for i in tqdm(range(0, num_chunks, batch_size), desc="Upserting batches"):
        batch_chunks = chunks_to_store[i:i + batch_size]
        vectors_to_upsert = []

        for chunk in batch_chunks:
            if 'embedding' not in chunk or not chunk['embedding']:
                logger.warning(f"Skipping chunk ID {chunk.get('id', 'N/A')} due to missing embedding.")
                continue

            pinecone_metadata = {"original_text": chunk.get('text', '')}
            for key, value in chunk.get('metadata', {}).items():
                if isinstance(value, (str, int, float, bool)):
                    pinecone_metadata[key] = value
                elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                    pinecone_metadata[key] = value
                else:
                    pinecone_metadata[key] = str(value)
                    logger.debug(f"Converted metadata key '{key}' to string.")

            vectors_to_upsert.append(
                (
                    chunk['id'],
                    chunk['embedding'],
                    pinecone_metadata
                )
            )

        if not vectors_to_upsert:
            logger.info(f"Batch {i // batch_size + 1} is empty. Skipping.")
            continue

        try:
            logger.debug(f"Upserting batch {i // batch_size + 1} with {len(vectors_to_upsert)} vectors...")
            pinecone_index.upsert(vectors=vectors_to_upsert)
        except Exception as e:
            logger.error(f"Error upserting batch starting at index {i}: {e}", exc_info=True)

    logger.info(f"Finished upserting {num_chunks} chunks.")
    try:
        stats_after = pinecone_index.describe_index_stats()
        logger.info(f"Index stats after upsert: {stats_after}")
    except Exception as e:
        logger.warning(f"Could not retrieve index stats: {e}")

    return True

def process_pdf_pipeline(pdf_path: str, document_metadata: Dict) -> Optional[List[Dict]]:
    """Main pipeline to process PDF and store chunks in Pinecone."""
    try:
        # Initialize environment
        project_id = initialize_environment()
        
        # Initialize clients
        mistral_client = initialize_mistral_client(api_key=os.environ.get("MISTRAL_API_KEY", "mistral_api_key"))
        vertex_model = initialize_vertex_ai(project_id)
        pinecone_index = initialize_pinecone(
            api_key=os.environ.get("PINECONE_API_KEY", "pinecone_api_key"),
            environment=os.environ.get("PINECONE_ENVIRONMENT", "us-east-1-aws"),
            index_name=os.environ.get("PINECONE_INDEX_NAME", "course-feedback-rag")
        )

        # Process PDF
        ocr_text = process_pdf_ocr(mistral_client, pdf_path)
        
        # Chunk text
        chunks = chunk_feedback_by_qa(ocr_text, document_metadata)
        
        # Add embeddings
        chunks_with_embeddings = add_embeddings_to_chunks(chunks, vertex_model)
        if not chunks_with_embeddings:
            logger.error("Failed to add embeddings.")
            return None
        
        # Store in Pinecone
        if pinecone_index:
            success = store_chunks_in_pinecone(
                chunks_with_embeddings,
                pinecone_index,
                index_name="course-feedback-rag"
            )
            if not success:
                logger.error("Failed to store chunks in Pinecone.")
                return None
        
        logger.info("PDF processing pipeline completed successfully.")
        return chunks_with_embeddings
    
    except Exception as e:
        logger.error(f"Error in PDF processing pipeline: {e}", exc_info=True)
        return None

# if __name__ == "__main__":
    doc_meta = {
        'instructor_name': 'Tejas Parikh',
        'course_code': 'CSYE 6225',
        'semester_term': 'Spring',
        'semester_year': '2024',
        'course_name': 'Network Structures and Cloud Computing',
        'credit_hours': '4',
        'bucket_path': 'gs://csye7125-trace-data/Parikh_Tejas_000937178_Spring-2024_CSYE622503Lecture_Comments-Report.pdf'
    }
    
    pdf_path = "/content/tejas_parikh_1.pdf"
    chunks_with_embeddings = process_pdf_pipeline(pdf_path, doc_meta)
    
    if chunks_with_embeddings:
        for i, chunk in enumerate(chunks_with_embeddings[:5]):
            print(f"--- Chunk {i+1} (with Embedding) ---")
            print(f"ID: {chunk['id']}")
            print(f"Text: {chunk['text']}")
            embedding_preview = chunk.get('embedding')
            if embedding_preview:
                print(f"Embedding (first 5 dims): {embedding_preview[:5]}...")
                print(f"Embedding dimensions: {len(embedding_preview)}")
            else:
                print("Embedding: Error or not generated.")
            print(f"Metadata: {chunk['metadata']}")
            print("-" * 20)