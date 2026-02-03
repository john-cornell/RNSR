"""Tests for the Table Parser system."""

import pytest

from rnsr.ingestion.table_parser import (
    ParsedTable,
    TableCell,
    TableColumn,
    TableParser,
    TableQueryEngine,
    TableRow,
    parse_tables_from_text,
    query_table,
)


class TestTableCell:
    """Tests for TableCell dataclass."""
    
    def test_cell_creation(self):
        """Test creating a table cell."""
        cell = TableCell(
            row=0,
            col=1,
            value="Test Value",
            is_header=True,
            data_type="text",
        )
        
        assert cell.row == 0
        assert cell.col == 1
        assert cell.value == "Test Value"
        assert cell.is_header is True
    
    def test_cell_to_dict(self):
        """Test converting cell to dictionary."""
        cell = TableCell(row=0, col=0, value="Test")
        
        data = cell.to_dict()
        
        assert data["value"] == "Test"
        assert "row" in data


class TestTableRow:
    """Tests for TableRow dataclass."""
    
    def test_row_creation(self):
        """Test creating a table row."""
        cells = [
            TableCell(row=0, col=0, value="A"),
            TableCell(row=0, col=1, value="B"),
        ]
        
        row = TableRow(index=0, cells=cells, is_header_row=True)
        
        assert row.index == 0
        assert len(row.cells) == 2
        assert row.is_header_row is True
    
    def test_row_get_values(self):
        """Test getting all values from a row."""
        cells = [
            TableCell(row=0, col=0, value="X"),
            TableCell(row=0, col=1, value="Y"),
            TableCell(row=0, col=2, value="Z"),
        ]
        
        row = TableRow(index=0, cells=cells)
        
        values = row.get_values()
        
        assert values == ["X", "Y", "Z"]


class TestParsedTable:
    """Tests for ParsedTable dataclass."""
    
    def test_table_creation(self):
        """Test creating a parsed table."""
        table = ParsedTable(
            doc_id="doc_123",
            title="Test Table",
            headers=["Name", "Value"],
            num_rows=3,
            num_cols=2,
        )
        
        assert table.doc_id == "doc_123"
        assert table.title == "Test Table"
        assert len(table.headers) == 2
    
    def test_table_get_cell(self):
        """Test getting a specific cell."""
        cells = [
            TableCell(row=0, col=0, value="A"),
            TableCell(row=0, col=1, value="B"),
        ]
        rows = [TableRow(index=0, cells=cells)]
        
        table = ParsedTable(rows=rows)
        
        cell = table.get_cell(0, 0)
        assert cell.value == "A"
        
        cell = table.get_cell(0, 1)
        assert cell.value == "B"
        
        cell = table.get_cell(5, 5)
        assert cell is None
    
    def test_table_to_markdown(self):
        """Test converting table to markdown."""
        cells1 = [
            TableCell(row=0, col=0, value="Name"),
            TableCell(row=0, col=1, value="Value"),
        ]
        cells2 = [
            TableCell(row=1, col=0, value="Item A"),
            TableCell(row=1, col=1, value="100"),
        ]
        
        rows = [
            TableRow(index=0, cells=cells1, is_header_row=True),
            TableRow(index=1, cells=cells2, is_header_row=False),
        ]
        
        table = ParsedTable(
            rows=rows,
            headers=["Name", "Value"],
        )
        
        markdown = table.to_markdown()
        
        assert "| Name | Value |" in markdown
        assert "| Item A | 100 |" in markdown
    
    def test_table_to_csv(self):
        """Test converting table to CSV."""
        cells1 = [
            TableCell(row=0, col=0, value="Name"),
            TableCell(row=0, col=1, value="Value"),
        ]
        cells2 = [
            TableCell(row=1, col=0, value="Item A"),
            TableCell(row=1, col=1, value="100"),
        ]
        
        rows = [
            TableRow(index=0, cells=cells1),
            TableRow(index=1, cells=cells2),
        ]
        
        table = ParsedTable(rows=rows)
        
        csv = table.to_csv()
        
        assert "Name,Value" in csv
        assert "Item A,100" in csv


class TestTableParser:
    """Tests for TableParser class."""
    
    def test_parser_creation(self):
        """Test creating a table parser."""
        parser = TableParser(
            infer_types=True,
            detect_headers=True,
        )
        
        assert parser.infer_types is True
        assert parser.detect_headers is True
    
    def test_parse_markdown_table(self):
        """Test parsing a markdown-style table."""
        text = """
        Some text before the table.
        
        | Name | Amount | Status |
        |------|--------|--------|
        | Item A | $100 | Active |
        | Item B | $200 | Pending |
        
        Some text after.
        """
        
        parser = TableParser()
        tables = parser.parse_from_text(text)
        
        assert len(tables) >= 1
        table = tables[0]
        assert table.num_cols == 3
        assert "Name" in table.headers
    
    def test_parse_multiple_tables(self):
        """Test parsing multiple tables from text."""
        text = """
        Table 1:
        | A | B |
        |---|---|
        | 1 | 2 |
        
        Some text.
        
        Table 2:
        | X | Y | Z |
        |---|---|---|
        | a | b | c |
        """
        
        parser = TableParser()
        tables = parser.parse_from_text(text)
        
        assert len(tables) >= 2
    
    def test_infer_number_type(self):
        """Test inferring number data type."""
        parser = TableParser(infer_types=True)
        
        data_type, value = parser._infer_cell_type("12345")
        assert data_type == "number"
        assert value == 12345.0
    
    def test_infer_currency_type(self):
        """Test inferring currency data type."""
        parser = TableParser(infer_types=True)
        
        data_type, value = parser._infer_cell_type("$1,234.56")
        assert data_type == "currency"
        assert value == 1234.56
    
    def test_infer_percentage_type(self):
        """Test inferring percentage data type."""
        parser = TableParser(infer_types=True)
        
        data_type, value = parser._infer_cell_type("75%")
        assert data_type == "percentage"
        assert value == 75.0
    
    def test_infer_date_type(self):
        """Test inferring date data type."""
        parser = TableParser(infer_types=True)
        
        data_type, value = parser._infer_cell_type("01/15/2024")
        assert data_type == "date"
    
    def test_infer_empty_type(self):
        """Test inferring empty data type."""
        parser = TableParser(infer_types=True)
        
        data_type, value = parser._infer_cell_type("")
        assert data_type == "empty"


class TestTableQueryEngine:
    """Tests for TableQueryEngine class."""
    
    @pytest.fixture
    def sample_table(self):
        """Create a sample table for testing."""
        header_cells = [
            TableCell(row=0, col=0, value="Name", is_header=True),
            TableCell(row=0, col=1, value="Amount", is_header=True),
            TableCell(row=0, col=2, value="Status", is_header=True),
        ]
        
        row1_cells = [
            TableCell(row=1, col=0, value="Item A", data_type="text"),
            TableCell(row=1, col=1, value="100", data_type="number", numeric_value=100.0),
            TableCell(row=1, col=2, value="Active", data_type="text"),
        ]
        
        row2_cells = [
            TableCell(row=2, col=0, value="Item B", data_type="text"),
            TableCell(row=2, col=1, value="200", data_type="number", numeric_value=200.0),
            TableCell(row=2, col=2, value="Pending", data_type="text"),
        ]
        
        row3_cells = [
            TableCell(row=3, col=0, value="Item C", data_type="text"),
            TableCell(row=3, col=1, value="150", data_type="number", numeric_value=150.0),
            TableCell(row=3, col=2, value="Active", data_type="text"),
        ]
        
        rows = [
            TableRow(index=0, cells=header_cells, is_header_row=True),
            TableRow(index=1, cells=row1_cells),
            TableRow(index=2, cells=row2_cells),
            TableRow(index=3, cells=row3_cells),
        ]
        
        return ParsedTable(
            rows=rows,
            headers=["Name", "Amount", "Status"],
            num_rows=4,
            num_cols=3,
        )
    
    def test_select_all(self, sample_table):
        """Test selecting all columns."""
        engine = TableQueryEngine(sample_table)
        
        results = engine.select()
        
        assert len(results) == 3  # 3 data rows
        assert "Name" in results[0]
        assert "Amount" in results[0]
    
    def test_select_specific_columns(self, sample_table):
        """Test selecting specific columns."""
        engine = TableQueryEngine(sample_table)
        
        results = engine.select(columns=["Name", "Status"])
        
        assert len(results) == 3
        assert "Name" in results[0]
        assert "Status" in results[0]
        assert "Amount" not in results[0]
    
    def test_select_with_where(self, sample_table):
        """Test selecting with WHERE filter."""
        engine = TableQueryEngine(sample_table)
        
        results = engine.select(where={"Status": "Active"})
        
        assert len(results) == 2
        assert all(r["Status"] == "Active" for r in results)
    
    def test_select_with_order_by(self, sample_table):
        """Test selecting with ORDER BY."""
        engine = TableQueryEngine(sample_table)
        
        results = engine.select(order_by="Name")
        
        assert len(results) == 3
        # Results should be sorted alphabetically by Name
    
    def test_select_with_limit(self, sample_table):
        """Test selecting with LIMIT."""
        engine = TableQueryEngine(sample_table)
        
        results = engine.select(limit=2)
        
        assert len(results) == 2
    
    def test_aggregate_sum(self, sample_table):
        """Test SUM aggregation."""
        engine = TableQueryEngine(sample_table)
        
        total = engine.aggregate("Amount", "sum")
        
        assert total == 450.0  # 100 + 200 + 150
    
    def test_aggregate_avg(self, sample_table):
        """Test AVG aggregation."""
        engine = TableQueryEngine(sample_table)
        
        avg = engine.aggregate("Amount", "avg")
        
        assert avg == 150.0  # 450 / 3
    
    def test_aggregate_count(self, sample_table):
        """Test COUNT aggregation."""
        engine = TableQueryEngine(sample_table)
        
        count = engine.aggregate("Amount", "count")
        
        assert count == 3
    
    def test_aggregate_max(self, sample_table):
        """Test MAX aggregation."""
        engine = TableQueryEngine(sample_table)
        
        max_val = engine.aggregate("Amount", "max")
        
        assert max_val == 200.0
    
    def test_aggregate_min(self, sample_table):
        """Test MIN aggregation."""
        engine = TableQueryEngine(sample_table)
        
        min_val = engine.aggregate("Amount", "min")
        
        assert min_val == 100.0
    
    def test_aggregate_with_where(self, sample_table):
        """Test aggregation with WHERE filter."""
        engine = TableQueryEngine(sample_table)
        
        total = engine.aggregate("Amount", "sum", where={"Status": "Active"})
        
        assert total == 250.0  # 100 + 150 (Active items only)
    
    def test_get_row_data(self, sample_table):
        """Test selecting rows returns expected data."""
        engine = TableQueryEngine(sample_table)
        
        results = engine.select(limit=1)
        
        assert len(results) == 1
        # First data row should have Name = "Item A"
        assert results[0].get("Name") == "Item A"


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_parse_tables_from_text(self):
        """Test parse_tables_from_text convenience function."""
        text = """
        | Col1 | Col2 |
        |------|------|
        | A | B |
        """
        
        tables = parse_tables_from_text(text)
        
        assert len(tables) >= 1
    
    def test_query_table(self):
        """Test query_table convenience function."""
        cells = [
            TableCell(row=0, col=0, value="Name", is_header=True),
            TableCell(row=0, col=1, value="Value", is_header=True),
        ]
        data_cells = [
            TableCell(row=1, col=0, value="A"),
            TableCell(row=1, col=1, value="1"),
        ]
        
        rows = [
            TableRow(index=0, cells=cells, is_header_row=True),
            TableRow(index=1, cells=data_cells),
        ]
        
        table = ParsedTable(rows=rows, headers=["Name", "Value"])
        
        results = query_table(table, columns=["Name"])
        
        assert len(results) == 1
        assert results[0]["Name"] == "A"
