from textual.app import App, ComposeResult
from textual.widgets import DataTable, RichLog, Header, Footer
from textual.containers import Vertical
import time

class DashboardApp(App):
    CSS = """
    DataTable {
        height: 1fr;
        border: solid cyan;
    }
    RichLog {
        height: 2fr;
        border: solid blue;
    }
    """
    
    def __init__(self, alpaca_client):
        super().__init__()
        self.alpaca = alpaca_client
        self.table = DataTable()
        self.log_widget = RichLog(highlight=True, markup=True, wrap=True)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Vertical(self.table, self.log_widget)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Ultimate AI Trading Bot"
        self.table.add_columns("Symbol", "Qty", "Market Value", "Unrealized P/L")
        self.update_table()
        # Update the table automatically every 5 seconds
        self.set_interval(5, self.update_table)

    def update_table(self):
        self.table.clear()
        try:
            positions = self.alpaca.get_positions()
            if not positions:
                self.table.add_row("No positions", "-", "-", "-")
            else:
                for pos in positions:
                    pl_color = "green" if float(pos.unrealized_pl) >= 0 else "red"
                    self.table.add_row(
                        pos.symbol,
                        str(pos.qty),
                        f"${float(pos.market_value):.2f}",
                        f"[{pl_color}]${float(pos.unrealized_pl):.2f}[/{pl_color}]"
                    )
        except Exception:
            self.table.add_row("Error fetching", "-", "-", "-")

    def log_message(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] {message}"
        
        # If the UI is fully loaded, write to the log widget securely from a background thread
        if self.is_running:
            self.call_from_thread(self.log_widget.write, formatted_msg)
        else:
            print(formatted_msg)
