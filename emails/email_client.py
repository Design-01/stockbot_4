import win32com.client
import os
from typing import List, Optional, Union
import pathlib
import pandas as pd
from datetime import datetime
import numpy as np

def send_outlook_email(
    subject: str,
    body: Union[str, dict],
    image_paths: Optional[List[str]] = None,
    recipients: Union[str, List[str]] = None,
    cc: Union[str, List[str]] = None,
    is_html: bool = True,
    importance: str = "Normal"
) -> bool:
    """
    Send an email using Outlook with optional HTML formatting and image attachments.
    
    Args:
        subject (str): Email subject line
        body (Union[str, dict]): Email body content. If is_html=True, should be HTML formatted string.
            If dict, expects {'text': str, 'embedded_images': List[dict]} where embedded_images contains
            {'path': str, 'cid': str} for each inline image
        image_paths (List[str], optional): List of image file paths to attach
        recipients (Union[str, List[str]], optional): Email recipient(s)
        cc (Union[str, List[str]], optional): CC recipient(s)
        is_html (bool): Whether body contains HTML formatting (default True)
        importance (str): Email importance - "Low", "Normal" or "High"
        
    Returns:
        bool: True if email sent successfully, False otherwise
        
    Example:
        # Simple text email
        send_outlook_email(
            "Test Subject",
            "Hello World",
            is_html=False
        )
        
        # HTML email with embedded image
        html_body = {
            'text': '''
                <h2 style="color: #2e6c80;">Monthly Report</h2>
                <p>Please see the chart below:</p>
                <img src="cid:chart1" width="500"><br>
                <p>Best regards,<br>Analysis Team</p>
            ''',
            'embedded_images': [
                {'path': 'path/to/chart.png', 'cid': 'chart1'}
            ]
        }
        send_outlook_email(
            "Monthly Analysis",
            html_body,
            image_paths=['path/to/attachment1.png']
        )
    """
    try:
        outlook = win32com.client.Dispatch('Outlook.Application')
        mail = outlook.CreateItem(0)  # 0 = olMailItem
        
        # Set recipients
        if recipients:
            if isinstance(recipients, str):
                mail.To = recipients
            else:
                mail.To = "; ".join(recipients)
        
        # Set CC
        if cc:
            if isinstance(cc, str):
                mail.CC = cc
            else:
                mail.CC = "; ".join(cc)
                
        # Set subject
        mail.Subject = subject
        
        # Set importance
        importance_levels = {"Low": 0, "Normal": 1, "High": 2}
        mail.Importance = importance_levels.get(importance, 1)
        
        # Handle body content
        if isinstance(body, dict) and is_html:
            # Handle HTML with embedded images
            mail.HTMLBody = body['text']
            if 'embedded_images' in body:
                for img in body['embedded_images']:
                    if os.path.exists(img['path']):
                        attachment = mail.Attachments.Add(
                            os.path.abspath(img['path']),
                            1,  # olByValue
                            0,  # Position
                            img['cid']
                        )
                        attachment.PropertyAccessor.SetProperty(
                            "http://schemas.microsoft.com/mapi/proptag/0x3712001F",
                            img['cid']
                        )
        else:
            # Handle simple text or HTML
            if is_html:
                mail.HTMLBody = body
            else:
                mail.Body = body
        
        # Add attachments
        if image_paths:
            for path in image_paths:
                if os.path.exists(path):
                    mail.Attachments.Add(os.path.abspath(path))
        
        mail.Send()
        return True
        
    except Exception as e:
        print(f"Error sending email: {str(e)}")
        return False


def style_and_save_dataframe(df: pd.DataFrame, 
                            output_path: str = "styled_table.html",
                            custom_style: bool = True,
                            max_rows: int = 1000,
                            max_cols: int = None) -> tuple[str, bool, bool]:
    """
    Style a DataFrame and save it as an HTML file with size handling.
    
    Args:
        df: Pandas DataFrame to style
        output_path: Path where to save the HTML file
        custom_style: Whether to apply custom styling
        max_rows: Maximum number of rows to display (None for all)
        max_cols: Maximum number of columns to display (None for all)
    
    Returns:
        tuple: (path to HTML file, whether rows were truncated, whether cols were truncated)
    """
    # Handle large DataFrames
    rows_truncated = False
    cols_truncated = False
    
    # Create a deep copy of the DataFrame to avoid warnings
    display_df = df.copy()
    
    if max_rows and len(display_df) > max_rows:
        display_df = display_df.iloc[:max_rows].copy()
        rows_truncated = True
        
    if max_cols and len(display_df.columns) > max_cols:
        display_df = display_df.iloc[:, :max_cols].copy()
        cols_truncated = True
    
    # Format numeric columns safely using loc
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if display_df[col].dtype != int:
            display_df.loc[:, col] = display_df[col].round(2)
    
    if custom_style:
        styled_df = display_df.style\
            .set_properties(**{
                'background-color': '#f5f5f5',
                'border-color': '#888888',
                'border-style': 'solid',
                'border-width': '1px',
                'padding': '5px',
                'text-align': 'left',
                'white-space': 'nowrap',
                'overflow': 'hidden',
                'text-overflow': 'ellipsis',
                'max-width': '300px'
            })\
            .set_table_styles([
                {'selector': 'th',
                 'props': [('background-color', '#007bff'),
                          ('color', 'white'),
                          ('font-weight', 'bold'),
                          ('padding', '5px'),
                          ('border', '1px solid #888888'),
                          ('text-align', 'left')]},
                {'selector': 'tr:nth-of-type(odd)',
                 'props': [('background-color', '#f9f9f9')]},
                {'selector': 'table',
                 'props': [('border-collapse', 'collapse'),
                          ('width', '100%'),
                          ('max-width', '100%'),
                          ('margin-bottom', '1rem'),
                          ('table-layout', 'auto')]}
            ])
        
        # Apply different styling to numeric columns
        for col in numeric_cols:
            styled_df = styled_df.set_properties(subset=[col], **{
                'text-align': 'right'
            })
    else:
        styled_df = display_df.style
    
    # Save to HTML
    styled_df.to_html(output_path)
    return output_path, rows_truncated, cols_truncated

def create_email_body_with_df(df: pd.DataFrame, 
                            include_summary: bool = True,
                            max_rows: int = 1000,
                            max_cols: int = None) -> str:
    """
    Create an email body with an embedded DataFrame.
    
    Args:
        df: DataFrame to include
        include_summary: Whether to include DataFrame summary
        max_rows: Maximum rows to display
        max_cols: Maximum columns to display
    
    Returns:
        str: Complete HTML email body
    """
    # Save styled DataFrame and get the path and truncation info
    result = style_and_save_dataframe(
        df, 
        output_path="temp_styled_table.html",
        max_rows=max_rows, 
        max_cols=max_cols
    )
    html_file = result[0]  # Extract just the file path
    rows_truncated = result[1]
    cols_truncated = result[2]
    
    # Read the styled table HTML
    with open(html_file, 'r') as f:
        table_html = f.read()
    
    # Create the email body
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    email_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif;">
        <h2 style="color: #2e6c80;">DataFrame Report</h2>
        <p>Generated on: {timestamp}</p>
    """
    
    if include_summary:
        email_body += f"""
        <h3>Data Summary:</h3>
        <ul>
            <li>Total Rows: {len(df):,}</li>
            <li>Total Columns: {len(df.columns)}</li>
            <li>Memory Usage: {df.memory_usage().sum() / 1024 / 1024:.2f} MB</li>
            <li>Data Types: {', '.join([f"{col}: {df[col].dtype}" for col in df.columns])}</li>
        </ul>
        """
        
        # Add warnings if data was truncated
        if rows_truncated or cols_truncated:
            email_body += '<div style="background-color: #fff3cd; padding: 10px; border: 1px solid #ffeeba; margin: 10px 0;">'
            if rows_truncated:
                email_body += f'<p style="color: #856404; margin: 0;">⚠️ Showing first {max_rows:,} rows of {len(df):,} total rows</p>'
            if cols_truncated:
                email_body += f'<p style="color: #856404; margin: 0;">⚠️ Showing first {max_cols} columns of {len(df.columns)} total columns</p>'
            email_body += '</div>'
    
    email_body += f"""
        {table_html}
        
        <p style="color: #666; font-size: 12px; margin-top: 20px;">
            This is an automated report. Please do not reply to this email.
        </p>
    </body>
    </html>
    """
    
    # Clean up the temporary HTML file
    os.remove(html_file)
    
    return email_body