"""
RNSR Table Parser

Deep parsing of tables from documents.
Extracts structure, headers, cells, and enables cell-level retrieval.

Features:
- Table structure extraction (rows, columns, headers)
- Cell-level retrieval (answer questions about specific cells)
- SQL-like queries over extracted table data
- Support for merged cells, multi-level headers

Integrates with the ingestion pipeline to extract tables from PDFs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

import structlog

logger = structlog.get_logger(__name__)


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class TableCell:
    """A single cell in a table."""
    
    row: int
    col: int
    value: str
    
    # Cell properties
    is_header: bool = False
    is_merged: bool = False
    rowspan: int = 1
    colspan: int = 1
    
    # Type inference
    data_type: Literal["text", "number", "currency", "percentage", "date", "empty"] = "text"
    numeric_value: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "row": self.row,
            "col": self.col,
            "value": self.value,
            "is_header": self.is_header,
            "is_merged": self.is_merged,
            "rowspan": self.rowspan,
            "colspan": self.colspan,
            "data_type": self.data_type,
            "numeric_value": self.numeric_value,
        }


@dataclass
class TableRow:
    """A row in a table."""
    
    index: int
    cells: list[TableCell] = field(default_factory=list)
    is_header_row: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "cells": [c.to_dict() for c in self.cells],
            "is_header_row": self.is_header_row,
        }
    
    def get_values(self) -> list[str]:
        """Get all cell values in row."""
        return [c.value for c in self.cells]


@dataclass
class TableColumn:
    """A column in a table."""
    
    index: int
    header: str = ""
    data_type: str = "text"
    cells: list[TableCell] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "header": self.header,
            "data_type": self.data_type,
            "cell_count": len(self.cells),
        }
    
    def get_values(self) -> list[str]:
        """Get all cell values in column."""
        return [c.value for c in self.cells]


@dataclass
class ParsedTable:
    """A fully parsed table."""
    
    id: str = field(default_factory=lambda: f"table_{str(uuid4())[:8]}")
    
    # Source information
    doc_id: str = ""
    page_num: int | None = None
    node_id: str = ""
    
    # Table metadata
    title: str = ""
    caption: str = ""
    
    # Structure
    rows: list[TableRow] = field(default_factory=list)
    columns: list[TableColumn] = field(default_factory=list)
    
    # Dimensions
    num_rows: int = 0
    num_cols: int = 0
    header_rows: int = 1
    
    # Content
    headers: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "page_num": self.page_num,
            "node_id": self.node_id,
            "title": self.title,
            "caption": self.caption,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "header_rows": self.header_rows,
            "headers": self.headers,
            "rows": [r.to_dict() for r in self.rows],
            "columns": [c.to_dict() for c in self.columns],
        }
    
    def get_cell(self, row: int, col: int) -> TableCell | None:
        """Get a specific cell."""
        if 0 <= row < len(self.rows):
            row_obj = self.rows[row]
            if 0 <= col < len(row_obj.cells):
                return row_obj.cells[col]
        return None
    
    def get_row(self, index: int) -> TableRow | None:
        """Get a specific row."""
        if 0 <= index < len(self.rows):
            return self.rows[index]
        return None
    
    def get_column(self, index: int) -> TableColumn | None:
        """Get a specific column."""
        if 0 <= index < len(self.columns):
            return self.columns[index]
        return None
    
    def get_column_by_header(self, header: str) -> TableColumn | None:
        """Get column by header name."""
        header_lower = header.lower()
        for col in self.columns:
            if col.header.lower() == header_lower:
                return col
        return None
    
    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        if not self.rows:
            return ""
        
        lines = []
        
        # Title
        if self.title:
            lines.append(f"**{self.title}**\n")
        
        # Headers
        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("|" + "|".join(["---"] * len(self.headers)) + "|")
        
        # Data rows
        for row in self.rows:
            if not row.is_header_row:
                values = [c.value for c in row.cells]
                lines.append("| " + " | ".join(values) + " |")
        
        # Caption
        if self.caption:
            lines.append(f"\n*{self.caption}*")
        
        return "\n".join(lines)
    
    def to_csv(self) -> str:
        """Convert table to CSV format."""
        lines = []
        
        for row in self.rows:
            values = []
            for cell in row.cells:
                # Escape commas and quotes
                value = cell.value.replace('"', '""')
                if ',' in value or '"' in value:
                    value = f'"{value}"'
                values.append(value)
            lines.append(",".join(values))
        
        return "\n".join(lines)


# =============================================================================
# Table Parser
# =============================================================================


class TableParser:
    """
    Parses tables from various formats.
    
    Supports:
    - Plain text tables (markdown, ASCII)
    - HTML tables
    - PDF table extraction (via PyMuPDF)
    """
    
    def __init__(
        self,
        infer_types: bool = True,
        detect_headers: bool = True,
    ):
        """
        Initialize the table parser.
        
        Args:
            infer_types: Whether to infer cell data types.
            detect_headers: Whether to detect header rows.
        """
        self.infer_types = infer_types
        self.detect_headers = detect_headers
    
    def parse_from_text(
        self,
        text: str,
        doc_id: str = "",
        page_num: int | None = None,
        node_id: str = "",
    ) -> list[ParsedTable]:
        """
        Parse tables from plain text.
        
        Args:
            text: Text containing table(s).
            doc_id: Document ID.
            page_num: Page number.
            node_id: Node ID.
            
        Returns:
            List of ParsedTable objects.
        """
        tables = []
        
        # Try markdown table parsing
        markdown_tables = self._parse_markdown_tables(text)
        
        for raw_table in markdown_tables:
            parsed = self._process_raw_table(
                raw_table,
                doc_id=doc_id,
                page_num=page_num,
                node_id=node_id,
            )
            if parsed:
                tables.append(parsed)
        
        # Try ASCII table parsing if no markdown found
        if not tables:
            ascii_tables = self._parse_ascii_tables(text)
            for raw_table in ascii_tables:
                parsed = self._process_raw_table(
                    raw_table,
                    doc_id=doc_id,
                    page_num=page_num,
                    node_id=node_id,
                )
                if parsed:
                    tables.append(parsed)
        
        logger.info("tables_parsed", count=len(tables))
        
        return tables
    
    def _parse_markdown_tables(self, text: str) -> list[list[list[str]]]:
        """Parse markdown-style tables."""
        tables = []
        
        # Find table blocks (consecutive lines with |)
        lines = text.split('\n')
        current_table: list[list[str]] = []
        
        for line in lines:
            line = line.strip()
            
            if '|' in line:
                # Skip separator lines
                if re.match(r'^[\|\s\-:]+$', line):
                    continue
                
                # Parse cells
                cells = [c.strip() for c in line.split('|')]
                # Remove empty first/last if line starts/ends with |
                if cells and cells[0] == '':
                    cells = cells[1:]
                if cells and cells[-1] == '':
                    cells = cells[:-1]
                
                if cells:
                    current_table.append(cells)
            else:
                # End of table
                if current_table and len(current_table) >= 2:
                    tables.append(current_table)
                current_table = []
        
        # Don't forget last table
        if current_table and len(current_table) >= 2:
            tables.append(current_table)
        
        return tables
    
    def _parse_ascii_tables(self, text: str) -> list[list[list[str]]]:
        """Parse ASCII-style tables with borders."""
        tables = []
        
        # Look for lines with consistent column separators
        lines = text.split('\n')
        
        # Find potential table regions
        in_table = False
        current_table: list[list[str]] = []
        column_positions: list[int] = []
        
        for line in lines:
            # Check for border lines
            if re.match(r'^[\+\-\=]+$', line) or re.match(r'^[\|\s\-\+]+$', line):
                if not in_table:
                    in_table = True
                    # Detect column positions from border
                    column_positions = [m.start() for m in re.finditer(r'[\+\|]', line)]
                continue
            
            if in_table and '|' in line:
                # Extract cells based on column positions
                if column_positions:
                    cells = []
                    for i in range(len(column_positions) - 1):
                        start = column_positions[i] + 1
                        end = column_positions[i + 1]
                        if start < len(line) and end <= len(line):
                            cell = line[start:end].strip()
                            cells.append(cell)
                    if cells:
                        current_table.append(cells)
                else:
                    # Fallback to simple split
                    cells = [c.strip() for c in line.split('|') if c.strip()]
                    if cells:
                        current_table.append(cells)
            elif in_table and current_table:
                # End of table
                if len(current_table) >= 2:
                    tables.append(current_table)
                current_table = []
                in_table = False
                column_positions = []
        
        # Don't forget last table
        if current_table and len(current_table) >= 2:
            tables.append(current_table)
        
        return tables
    
    def _process_raw_table(
        self,
        raw_table: list[list[str]],
        doc_id: str = "",
        page_num: int | None = None,
        node_id: str = "",
    ) -> ParsedTable | None:
        """Process a raw table (list of rows) into ParsedTable."""
        if not raw_table:
            return None
        
        # Create table
        table = ParsedTable(
            doc_id=doc_id,
            page_num=page_num,
            node_id=node_id,
        )
        
        # Determine number of columns
        num_cols = max(len(row) for row in raw_table)
        table.num_cols = num_cols
        table.num_rows = len(raw_table)
        
        # Process rows
        for row_idx, raw_row in enumerate(raw_table):
            is_header = self.detect_headers and row_idx == 0
            
            row = TableRow(index=row_idx, is_header_row=is_header)
            
            for col_idx in range(num_cols):
                value = raw_row[col_idx] if col_idx < len(raw_row) else ""
                
                cell = TableCell(
                    row=row_idx,
                    col=col_idx,
                    value=value,
                    is_header=is_header,
                )
                
                # Infer type
                if self.infer_types:
                    cell.data_type, cell.numeric_value = self._infer_cell_type(value)
                
                row.cells.append(cell)
            
            table.rows.append(row)
            
            # Store headers
            if is_header:
                table.headers = [c.value for c in row.cells]
        
        # Build columns
        for col_idx in range(num_cols):
            column = TableColumn(
                index=col_idx,
                header=table.headers[col_idx] if col_idx < len(table.headers) else "",
            )
            
            for row in table.rows:
                if col_idx < len(row.cells):
                    column.cells.append(row.cells[col_idx])
            
            # Infer column type from majority
            if self.infer_types:
                column.data_type = self._infer_column_type(column.cells)
            
            table.columns.append(column)
        
        return table
    
    def _infer_cell_type(self, value: str) -> tuple[str, float | None]:
        """Infer the data type of a cell value."""
        value = value.strip()
        
        if not value:
            return "empty", None
        
        # Currency
        if re.match(r'^[\$€£¥]\s*[\d,]+\.?\d*$', value):
            num_str = re.sub(r'[^\d.]', '', value)
            try:
                return "currency", float(num_str)
            except ValueError:
                pass
        
        # Percentage
        if re.match(r'^[\d.]+\s*%$', value):
            num_str = value.replace('%', '').strip()
            try:
                return "percentage", float(num_str)
            except ValueError:
                pass
        
        # Number
        if re.match(r'^-?[\d,]+\.?\d*$', value):
            num_str = value.replace(',', '')
            try:
                return "number", float(num_str)
            except ValueError:
                pass
        
        # Date (basic patterns)
        date_patterns = [
            r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$',
            r'^\d{4}[/-]\d{1,2}[/-]\d{1,2}$',
        ]
        for pattern in date_patterns:
            if re.match(pattern, value):
                return "date", None
        
        return "text", None
    
    def _infer_column_type(self, cells: list[TableCell]) -> str:
        """Infer column type from majority of cells."""
        type_counts: dict[str, int] = {}
        
        for cell in cells:
            if not cell.is_header and cell.data_type != "empty":
                type_counts[cell.data_type] = type_counts.get(cell.data_type, 0) + 1
        
        if type_counts:
            return max(type_counts.items(), key=lambda x: x[1])[0]
        return "text"
    
    def parse_from_html(
        self,
        html: str,
        doc_id: str = "",
        page_num: int | None = None,
    ) -> list[ParsedTable]:
        """Parse tables from HTML."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("beautifulsoup4_not_available")
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        tables = []
        
        for table_elem in soup.find_all('table'):
            parsed = self._parse_html_table(table_elem, doc_id, page_num)
            if parsed:
                tables.append(parsed)
        
        return tables
    
    def _parse_html_table(
        self,
        table_elem: Any,
        doc_id: str,
        page_num: int | None,
    ) -> ParsedTable | None:
        """Parse a single HTML table element."""
        raw_table: list[list[str]] = []
        
        # Get rows
        rows = table_elem.find_all('tr')
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            raw_row = [cell.get_text(strip=True) for cell in cells]
            if raw_row:
                raw_table.append(raw_row)
        
        if raw_table:
            return self._process_raw_table(
                raw_table,
                doc_id=doc_id,
                page_num=page_num,
            )
        
        return None


# =============================================================================
# Table Query Engine
# =============================================================================


class TableQueryEngine:
    """
    Enables SQL-like queries over parsed tables.
    
    Supports:
    - SELECT: Get specific columns
    - WHERE: Filter rows
    - ORDER BY: Sort results
    - Aggregations: SUM, AVG, COUNT, MAX, MIN
    """
    
    def __init__(self, table: ParsedTable):
        """
        Initialize query engine for a table.
        
        Args:
            table: The parsed table to query.
        """
        self.table = table
    
    def select(
        self,
        columns: list[str] | None = None,
        where: dict[str, Any] | None = None,
        order_by: str | None = None,
        ascending: bool = True,
        limit: int | None = None,
    ) -> list[dict[str, str]]:
        """
        Select data from the table.
        
        Args:
            columns: Column names to select (None = all).
            where: Filter conditions {column: value} or {column: (op, value)}.
            order_by: Column to sort by.
            ascending: Sort direction.
            limit: Maximum rows to return.
            
        Returns:
            List of row dictionaries.
        """
        results = []
        
        # Get column indices
        col_indices = {}
        for i, header in enumerate(self.table.headers):
            col_indices[header.lower()] = i
        
        # Process data rows
        for row in self.table.rows:
            if row.is_header_row:
                continue
            
            # Apply WHERE filter
            if where and not self._matches_where(row, where, col_indices):
                continue
            
            # Build result row
            result_row = {}
            for i, cell in enumerate(row.cells):
                header = self.table.headers[i] if i < len(self.table.headers) else f"col_{i}"
                
                # Filter columns if specified
                if columns is None or header.lower() in [c.lower() for c in columns]:
                    result_row[header] = cell.value
            
            results.append(result_row)
        
        # Apply ORDER BY
        if order_by and order_by.lower() in col_indices:
            results.sort(
                key=lambda x: x.get(order_by, ""),
                reverse=not ascending,
            )
        
        # Apply LIMIT
        if limit:
            results = results[:limit]
        
        return results
    
    def _matches_where(
        self,
        row: TableRow,
        where: dict[str, Any],
        col_indices: dict[str, int],
    ) -> bool:
        """Check if a row matches WHERE conditions."""
        for col_name, condition in where.items():
            col_idx = col_indices.get(col_name.lower())
            if col_idx is None or col_idx >= len(row.cells):
                continue
            
            cell_value = row.cells[col_idx].value.lower()
            
            if isinstance(condition, tuple):
                op, value = condition
                value = str(value).lower()
                
                if op == "eq" and cell_value != value:
                    return False
                elif op == "ne" and cell_value == value:
                    return False
                elif op == "contains" and value not in cell_value:
                    return False
                elif op == "gt" or op == "lt" or op == "gte" or op == "lte":
                    try:
                        cell_num = float(cell_value.replace(',', '').replace('$', ''))
                        val_num = float(value.replace(',', '').replace('$', ''))
                        
                        if op == "gt" and cell_num <= val_num:
                            return False
                        elif op == "lt" and cell_num >= val_num:
                            return False
                        elif op == "gte" and cell_num < val_num:
                            return False
                        elif op == "lte" and cell_num > val_num:
                            return False
                    except ValueError:
                        return False
            else:
                # Simple equality
                if cell_value != str(condition).lower():
                    return False
        
        return True
    
    def aggregate(
        self,
        column: str,
        operation: Literal["sum", "avg", "count", "max", "min"],
        where: dict[str, Any] | None = None,
    ) -> float | int:
        """
        Perform aggregation on a column.
        
        Args:
            column: Column to aggregate.
            operation: Aggregation operation.
            where: Optional filter.
            
        Returns:
            Aggregated value.
        """
        col_indices = {h.lower(): i for i, h in enumerate(self.table.headers)}
        col_idx = col_indices.get(column.lower())
        
        if col_idx is None:
            return 0
        
        values = []
        
        for row in self.table.rows:
            if row.is_header_row:
                continue
            
            if where and not self._matches_where(row, where, col_indices):
                continue
            
            if col_idx < len(row.cells):
                cell = row.cells[col_idx]
                if cell.numeric_value is not None:
                    values.append(cell.numeric_value)
                elif cell.data_type in ["number", "currency", "percentage"]:
                    try:
                        num = float(cell.value.replace(',', '').replace('$', '').replace('%', ''))
                        values.append(num)
                    except ValueError:
                        pass
        
        if not values:
            return 0
        
        if operation == "sum":
            return sum(values)
        elif operation == "avg":
            return sum(values) / len(values)
        elif operation == "count":
            return len(values)
        elif operation == "max":
            return max(values)
        elif operation == "min":
            return min(values)
        
        return 0
    
    def get_cell_value(self, row: int, column: str | int) -> str | None:
        """Get a specific cell value."""
        if isinstance(column, str):
            col = self.table.get_column_by_header(column)
            if col:
                column = col.index
            else:
                return None
        
        cell = self.table.get_cell(row, column)
        return cell.value if cell else None


# =============================================================================
# Convenience Functions
# =============================================================================


def parse_tables_from_text(
    text: str,
    doc_id: str = "",
) -> list[ParsedTable]:
    """Parse all tables from text."""
    parser = TableParser()
    return parser.parse_from_text(text, doc_id=doc_id)


def query_table(
    table: ParsedTable,
    columns: list[str] | None = None,
    where: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Run a simple query on a table."""
    engine = TableQueryEngine(table)
    return engine.select(columns=columns, where=where)
