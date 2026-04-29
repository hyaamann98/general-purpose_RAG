import argparse 
from src import DataLoad, MakeChunk, Embedding, ChromaStore

def MakeParser(parser):
    parser.add_argument(
        '--source', type=str, default="local")
    parser.add_argument(
        '--path', type=str)
    parser.add_argument(
        '--chunk_size', type=int, default=500)
    parser.add_argument(
        '--chunk_overlap', type=int, default=50)
    parser.add_argument(
        '--embed_model', type=str, default="intfloat/multilingual-e5-base")
    args = parser.parse_args()
    return args

def run(args):
    
    print("> Load Documents")
    dataLoad = DataLoad(args.source)
    text_list = dataLoad.load(args.path)

    print("> Make Chunks")
    make_chunk = MakeChunk(args.chunk_size,args.chunk_overlap)
    chunks = make_chunk.text_split(text_list)

    print("> Embedding")
    embedding = Embedding(args.embed_model)
    embedded_list = embedding.embed_documents(chunks)

    print("> Save DB")
    db_store =  ChromaStore()
    db_store.add(chunks, embedded_list)

    print("=== Complete Update DB ===")

if __name__ == "__main__":

    # parser
    parser = argparse.ArgumentParser()
    args = MakeParser(parser)
    # print(args.chunk_size)
    # print(args.chunk_overlap)

    # pipe line
    run(args)