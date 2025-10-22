import asyncio
import os
from enum import Enum
from typing import IO, Any, Dict, Iterable, List, Union

from dotenv import load_dotenv
from llama_cloud import ExtractConfig, ExtractMode
from llama_cloud_services import LlamaExtract, SourceText
from pydantic import BaseModel, Field

load_dotenv()


class IncomeStatement(BaseModel):
    company_name: str = Field(
        description="The name of the company"
    )
    total_revenue: float = Field(description="Total revenue amount")
    cost_of_goods_sold: float = Field(description="Cost of goods sold amount")
    gross_profit: float = Field(description="Gross profit amount")
    payroll_expense: float = Field(description="Payroll expense amount")
    utilities_expense: float = Field(description="Utilities expense amount")
    rent_expense: float = Field(description="Rent expense amount")
    depreciation_expense: float = Field(description="Depreciation expense amount")
    total_operating_expenses: float = Field(
        description="Total operating expenses amount"
    )
    interest_expense: float = Field(description="Interest expense amount")
    taxes: float = Field(description="Taxes amount")
    net_profit: float = Field(description="Net profit or net income amount")


def get_extraction_agent(
    agent_name: str = "income-statement-parser",
    data_schema: BaseModel = IncomeStatement,
) -> Any:  # LlamaExtractAgent is not exported, so using Any
    """
    Retrieves a LlamaExtract agent by name, or creates it if it doesn't exist.
    This is cached to avoid recreating the agent on every run.
    """
    extractor = LlamaExtract()
    try:
        # Check if an agent with this name already exists
        agent = extractor.get_agent(name=agent_name)
        return agent
    except Exception:  # Broad exception because the specific one isn't documented
        # If not, create a new one
        config = ExtractConfig(
            use_reasoning=True,
            cite_sources=True,
            extraction_mode=ExtractMode.MULTIMODAL,
        )
        agent = extractor.create_agent(
            name=agent_name, data_schema=data_schema, config=config
        )
        return agent


def _prepare_source(source: Union[str, IO[bytes]]) -> Union[str, SourceText]:
    """Prepares a source for extraction, handling both file paths and byte streams."""
    if isinstance(source, str):
        return source

    # For file-like objects (e.g., from Streamlit's st.file_uploader)
    filename = getattr(source, "name", "unknown_file")
    source.seek(0)  # Reset file pointer to the beginning
    file_bytes = source.read()
    return SourceText(file=file_bytes, filename=filename)


async def extract_documents(
    sources: Iterable[Union[str, IO[bytes]]],
    agent_name: str = "income-statement-parser",
) -> List[Dict[str, Any]]:
    """
    Asynchronously extracts structured data from a list of documents.
    Args:
        sources: An iterable of sources. Each source can be a file path (str)
                 or a file-like object with a .read() method (e.g., from Streamlit).
        agent_name: The name of the extraction agent to use.
    Returns:
        A list of dictionaries, each containing the extracted data and metadata
        for one document.
    """
    agent = get_extraction_agent(agent_name=agent_name, data_schema=IncomeStatement)

    tasks = []
    for source in sources:
        prepared_source = _prepare_source(source)
        tasks.append(agent.aextract(prepared_source))

    results = await asyncio.gather(*tasks)

    outputs = []
    for source, result in zip(sources, results):
        filename = getattr(
            source, "name", source if isinstance(source, str) else "unknown_file"
        )
        
        # Handle both Pydantic model and dict responses
        if hasattr(result.data, 'model_dump'):
            data = result.data.model_dump()
        elif isinstance(result.data, dict):
            data = result.data
        else:
            data = dict(result.data)
        
        output = {
            "filename": filename,
            "data": data,
            "metadata": result.extraction_metadata,
        }
        outputs.append(output)

    return outputs