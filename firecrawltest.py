from dotenv import load_dotenv
from langchain_community.document_loaders import FireCrawlLoader
load_dotenv()

def test():
    loader = FireCrawlLoader(
        url="https://firecrawl.dev",
        mode="crawl",
    )

    data = loader.load()

    print(data[0].page_content[:100])
    print(data[0].metadata)

if __name__ == "__main__":
    test()