import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional


class StockSummaryEmail:
    def __init__(self):
        self.css_styles = """
        <style>
            /* Base Container */
            .container {
                font-family: Arial, Helvetica, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #ffffff;
            }

            /* Header Title */
            h1 {
                font-size: 24px;
                font-weight: bold;
                margin: 0 0 20px 0;
                padding: 0;
                color: #000000;
            }

            /* Status Steps Section */
            .steps-table {
                width: 100%;
                margin: 30px 0;
                border-collapse: separate;
                border-spacing: 10px 0;
            }

            .step-circle {
                background-color: #4CAF50;
                color: #ffffff;
                width: 40px;
                height: 40px;
                line-height: 40px;
                border-radius: 50%;
                margin: 0 auto 12px auto;
                text-align: center;
                font-size: 16px;
                font-weight: normal;
            }

            .step-label {
                font-weight: bold;
                margin-bottom: 4px;
                line-height: 1.4;
                font-size: 14px;
                color: #000000;
            }

            .step-count {
                font-size: 16px;
                line-height: 1.4;
                color: #000000;
            }


            /* Section Titles */
            .section-title {
                color: #1a73e8;
                font-size: 18px;
                font-weight: bold;
                margin: 30px 0 15px 0;
                padding: 0 0 5px 0;
                border-bottom: 2px solid #e41e31;
                line-height: 1.4;
            }

            /* Common Table Styles */
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
                background-color: #ffffff;
            }

            th {
                background-color: #f8f9fa;
                font-weight: bold;
                padding: 12px;
                text-align: left;
                border: 1px solid #dddddd;
                white-space: nowrap;
                color: #000000;
            }

            td {
                padding: 12px;
                text-align: left;
                border: 1px solid #dddddd;
                color: #000000;
            }

            /* Validation Section */
            .validation-section {
                margin: 0 0 30px 0;
                padding: 15px;
                background-color: #f8f9fa;
                border: 1px solid #dddddd;
                border-radius: 5px;
            }

            .validation-title {
                font-size: 16px;
                font-weight: bold;
                margin: 0 0 10px 0;
                padding: 0;
                color: #000000;
            }

            .passed-column {
                border-right: 2px solid #666666 !important;
                background-color: #f8f9fa;
                font-weight: bold;
            }

            /* Scan Settings Styles */
            .scan-settings-table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }

            .scan-settings-table tr:nth-child(odd) {
                background-color: #f8f9fa;
            }

            .scan-settings-table td {
                padding: 12px;
                border: 1px solid #dddddd;
            }

            .scan-settings-table td:first-child {
                font-weight: 600;
                width: 33%;
            }

            .scan-settings-total {
                background-color: #e8f4ff;
                padding: 12px;
                text-align: center;
                font-weight: 600;
                border-radius: 4px;
                margin-top: 10px;
                border: 1px solid #dddddd;
            }

            .scan-timestamp {
                color: #666666;
                font-size: 14px;
                margin: 0 0 10px 0;
                padding: 0;
            }

            /* Status Styling */
            .positive {
                color: #28a745 !important;
            }

            .negative {
                color: #dc3545 !important;
            }

            .pending-row {
                background-color: #fff3cd;
            }

            .completed-row {
                background-color: #f8f9fa;
            }

            .in-trade-row {
                background-color: #d4edda;
            }

            .true-cell {
                background-color: #d4edda;
            }

            .false-cell {
                background-color: #f8d7da;
            }

            /* Additional email-specific resets */
            * {
                -webkit-text-size-adjust: 100%;
                -ms-text-size-adjust: 100%;
            }

            table, td {
                mso-table-lspace: 0pt;
                mso-table-rspace: 0pt;
            }

            img {
                -ms-interpolation-mode: bicubic;
            }
        </style>
        """

    def _format_number(self, value: float, include_sign: bool = True) -> str:
        """Format numbers with appropriate styling and signs."""
        if pd.isna(value):
            return "-"
        if isinstance(value, (int, float)):
            formatted = f"{value:,.2f}"
            if include_sign and value > 0:
                formatted = f"+{formatted}"
            return formatted
        return str(value)

    def _create_status_steps(self, status_dict: Dict[str, int]) -> str:
        """Create the status steps section with proper horizontal layout."""
        steps = [
            ("Sector Analysis", status_dict.get("sector_analysis", 0)),
            ("Stocks Scanned", status_dict.get("stocks_scanned", 0)),
            ("Stocks Validated", status_dict.get("stocks_validated", 0)),
            ("Stocks Scored", status_dict.get("stocks_scored", 0))
        ]
        
        html = """
        <table class="steps-table" cellpadding="0" cellspacing="0">
            <tr>
        """
        
        for i, (label, value) in enumerate(steps, 1):
            html += f"""
                <td align="center" style="vertical-align: top;">
                    <div class="step-circle">{i}</div>
                    <div class="step-label">{label}</div>
                    <div class="step-count">{value}</div>
                </td>
            """
        
        html += """
            </tr>
        </table>
        """
        return html

    def _create_activity_table(self, activity_df: pd.DataFrame) -> str:
        """Create the activity table with appropriate styling."""
        html = '<table>'
        html += '''
        <tr>
            <th>Stock Name</th>
            <th>Symbol</th>
            <th>TA Score</th>
            <th>Status</th>
            <th>Daily P&L</th>
            <th>Daily P&L %</th>
        </tr>
        '''
        
        total_pnl = 0
        total_pnl_pct = 0
        
        for _, row in activity_df.iterrows():
            row_class = f"{row['Status'].lower()}-row"
            pnl_class = 'positive' if row.get('Daily P&L', 0) > 0 else 'negative'
            
            html += f'''
            <tr class="{row_class}">
                <td>{row['Stock Name']}</td>
                <td>{row['Symbol']}</td>
                <td>{row['TA Score']}</td>
                <td>{row['Status']}</td>
                <td class="{pnl_class}">{self._format_number(row.get('Daily P&L', 0))}</td>
                <td class="{pnl_class}">{self._format_number(row.get('Daily P&L %', 0))}%</td>
            </tr>
            '''
            
            if not pd.isna(row.get('Daily P&L')):
                total_pnl += row.get('Daily P&L', 0)
                total_pnl_pct += row.get('Daily P&L %', 0)
        
        # Add totals row
        total_class = 'positive' if total_pnl > 0 else 'negative'
        html += f'''
        <tr>
            <td colspan="4"><strong>Totals</strong></td>
            <td class="{total_class}"><strong>{self._format_number(total_pnl)}</strong></td>
            <td class="{total_class}"><strong>{self._format_number(total_pnl_pct)}%</strong></td>
        </tr>
        '''
        
        html += '</table>'
        return html

    def _create_sector_etf_table(self, etf_df: pd.DataFrame) -> str:
        """Create the sector ETF table with appropriate styling."""
        html = '<div class="section-title">Sector ETFs</div>'
        html += '<div class="filters">Active Filters: ' + \
                'Outperforming SPY = "Yes", Overall Score > 5</div>'
        html += '<table>'
        
        # Add headers
        html += '<tr>'
        for col in etf_df.columns:
            html += f'<th>{col}</th>'
        html += '</tr>'
        
        # Add rows
        for _, row in etf_df.iterrows():
            html += '<tr>'
            for col in etf_df.columns:
                value = row[col]
                cell_class = ''
                
                if isinstance(value, (int, float)):
                    cell_class = 'positive' if value > 0 else 'negative'
                elif col in ['Momentum Direction', 'Volume Trend']:
                    cell_class = 'positive' if value == 'Positive' else 'negative'
                
                html += f'<td class="{cell_class}">{value}</td>'
            html += '</tr>'
        html += '</table>'
        return html

    def _create_validation_tables(self, fundamentals_df: pd.DataFrame, ta_df: pd.DataFrame) -> str:
        """Create the validation tables with updated styling and layout."""
        html = '<div class="section-title">Stocks Validated</div>'
        html += '<div class="validation-section">'
        
        # Fundamentals table
        html += '<div class="validation-title">Validated Fundamentals</div>'
        html += '<table>'
        
        # Reorganize columns to put 'Passed' first
        cols = ['Rank', 'Symbol', 'Fundamentals Passed'] + [col for col in fundamentals_df.columns 
                if col not in ['Rank', 'Symbol', 'Fundamentals Passed']]
        
        # Headers
        html += '<tr>'
        for col in cols:
            class_name = 'passed-column' if 'Passed' in col else ''
            html += f'<th class="{class_name}">{col}</th>'
        html += '</tr>'
        
        # Data rows
        for _, row in fundamentals_df.iterrows():
            html += '<tr>'
            for col in cols:
                value = row[col]
                cell_class = ''
                if 'Passed' in col:
                    cell_class = 'passed-column'
                if isinstance(value, bool):
                    cell_class += ' ' + ('true-cell' if value else 'false-cell')
                    value = '✓' if value else '✗'
                html += f'<td class="{cell_class}">{value}</td>'
            html += '</tr>'
        html += '</table>'
        
        # Technical Analysis table
        html += '<div class="validation-title">Validated TA</div>'
        html += self._create_ta_table(ta_df, is_validation=True)
        html += '</div>'
        return html

    def _create_ta_table(self, ta_df: pd.DataFrame, is_validation: bool = False) -> str:
        """Create a technical analysis table with updated styling."""
        html = '<table>'
        
        # Reorganize columns to put 'Passed' first
        cols = ['Rank', 'Symbol', 'TA Passed'] + [col for col in ta_df.columns 
                if col not in ['Rank', 'Symbol', 'TA Passed']]
        
        # Headers
        html += '<tr>'
        for col in cols:
            class_name = 'passed-column' if 'Passed' in col else ''
            html += f'<th class="{class_name}">{col}</th>'
        html += '</tr>'
        
        # Data rows
        for _, row in ta_df.iterrows():
            html += '<tr>'
            for col in cols:
                value = row[col]
                cell_class = ''
                if 'Passed' in col:
                    cell_class = 'passed-column'
                if isinstance(value, bool):
                    cell_class += ' ' + ('true-cell' if value else 'false-cell')
                    value = '✓' if value else '✗'
                html += f'<td class="{cell_class}">{value}</td>'
            html += '</tr>'
        html += '</table>'
        return html

    def _create_stock_scores(self, daily_ta_df: pd.DataFrame) -> str:
        """Create the stock scores section."""
        html = '<div class="section-title">Stock Scores</div>'
        html += '<div class="validation-title">Daily TA Scores</div>'
        html += self._create_ta_table(daily_ta_df, is_validation=False)
        return html
    
    def _format_volume(self, value: int) -> str:
        """Format volume numbers to K/M/B notation."""
        if value >= 1_000_000_000:
            return f"{value/1_000_000_000:.1f}B"
        elif value >= 1_000_000:
            return f"{value/1_000_000:.1f}M"
        elif value >= 1_000:
            return f"{value/1_000:.0f}K"
        return str(value)

    def _format_market_cap(self, value: int) -> str:
        """Format market cap numbers to M/B notation."""
        if value >= 1_000_000_000:
            return f"{value/1_000_000_000:.1f}B"
        else:
            return f"{value/1_000_000:.0f}M"

    def _format_price_range(self, min_price: float, max_price: float) -> str:
        """Format price range with dollar signs."""
        return f"${min_price:.2f} - ${max_price:.2f}"

    def _create_scan_settings(self, scan_settings: Dict) -> str:
        """Create the scan settings section with formatted values."""
        timestamp = datetime.strptime(scan_settings['timestamp'], '%Y-%m-%d %H:%M:%S')
        
        # Format the price range from raw numbers
        price_range = self._format_price_range(
            float(scan_settings['price_min']), 
            float(scan_settings['price_max'])
        )
        
        # Format volume and market cap from raw numbers
        volume = self._format_volume(int(scan_settings['volume']))
        market_cap = self._format_market_cap(int(scan_settings['market_cap']))
        
        html = f'''
        <div class="section-title">Stock Scanned</div>
        <div class="scan-timestamp">({scan_settings['timestamp']})</div>
        <table class="scan-settings-table">
            <tbody>
                <tr>
                    <td>Scan Code</td>
                    <td>{scan_settings['scan_code']}</td>
                </tr>
                <tr>
                    <td>Price Range</td>
                    <td>{price_range}</td>
                </tr>
                <tr>
                    <td>Volume</td>
                    <td>{volume}</td>
                </tr>
                <tr>
                    <td>Market Cap</td>
                    <td>{market_cap}</td>
                </tr>
                <tr>
                    <td>Change %</td>
                    <td>{float(scan_settings['change_percent']):.2f}</td>
                </tr>
            </tbody>
        </table>
        <div class="scan-settings-total">
            Total Stocks Scanned: {scan_settings['total_scanned']}
        </div>
        '''
        return html

    def generate_email(self, 
                        status_dict: Optional[Dict[str, int]] = None,
                        scan_settings: Optional[Dict] = None,
                        activity_df: Optional[pd.DataFrame] = None,
                        sector_etf_df: Optional[pd.DataFrame] = None,
                        fundamentals_df: Optional[pd.DataFrame] = None,
                        ta_df: Optional[pd.DataFrame] = None,
                        daily_ta_df: Optional[pd.DataFrame] = None) -> str:
            
            """Generate the complete email HTML."""
            status_steps = self._create_status_steps(status_dict) if status_dict is not None else "No data/n"
            activity_table = self._create_activity_table(activity_df) if activity_df is not None else "No data/n"
            scan_settings_table = self._create_scan_settings(scan_settings) if scan_settings is not None else "No data/n"
            sector_etf_table = self._create_sector_etf_table(sector_etf_df) if sector_etf_df is not None else "No data/n"
            validation_tables = self._create_validation_tables(fundamentals_df, ta_df) if fundamentals_df is not None and ta_df is not None else "No data/n"
            stock_scores = self._create_stock_scores(daily_ta_df) if daily_ta_df is not None else "No data/n"

            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                {self.css_styles}
            </head>
            <body>
                <div class="container">
                    <h1>Daily Stockbot Summary</h1>
                    {status_steps}
                    <h2>Activity</h2>
                    {activity_table}
                    {scan_settings_table}
                    {sector_etf_table}
                    {validation_tables}
                    {stock_scores}
                </div>
            </body>
            </html>
            """
            return html