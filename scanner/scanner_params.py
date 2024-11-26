from xml.etree import ElementTree as ET
import json
from typing import Dict, List, Any
import markdown
from pathlib import Path
import re

def save_scanner_params_as_xml_from_ib(ib, file_path: str = 'ib_scan_params.xml') -> str:
    """Save all available scanner codes to a local XML file"""
    # Get scanner parameters as string
    params = ib.reqScannerParameters()
    
    # Save the raw string to a local XML file
    with open(file_path, "w") as file:
        file.write(params)
    
    print("Scanner parameters have been saved to ib_scan_params.xml")
    
    return params


class ScannerDocsGenerator:
    def __init__(self, xml_file_path: str):
        # Read and clean the XML content
        with open(xml_file_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        # Remove any content before the XML declaration
        xml_content = re.sub(r'^.*?(?=<\?xml)', '', xml_content, flags=re.DOTALL)
        self.root = ET.fromstring(xml_content)
        self.scan_types = {}
        self.instrument_types = set()
        self.location_codes = set()
        self.available_filters = {}
        
    def safe_get_text(self, element, path, default=''):
        """Safely get text from an XML element."""
        if element is None:
            return default
        found = element.find(path)
        return found.text if found is not None else default
        
    def extract_scan_types(self):
        """Extract all available scan types and their descriptions."""
        for scan_type in self.root.findall(".//ScanCode"):  # Changed from ScanType
            code = self.safe_get_text(scan_type, 'code')
            display_name = self.safe_get_text(scan_type, 'displayName', code)
            if code:
                self.scan_types[code] = display_name
            
    def extract_locations(self):
        """Extract all available location codes from the LocationTree."""
        for location in self.root.findall(".//Location"):
            location_code = self.safe_get_text(location, 'locationCode')
            if location_code:
                self.location_codes.add(location_code)
                
    def extract_instruments(self):
        """Extract all available instrument types."""
        # Try direct instrument types first
        for instrument in self.root.findall(".//Instrument/type"):
            if instrument.text:
                self.instrument_types.add(instrument.text)
        
        # Also check 'instruments' tags
        for instrument in self.root.findall(".//instruments"):
            if instrument.text:
                self.instrument_types.add(instrument.text)
                
    def extract_filters(self):
        """Extract all available filter parameters."""
        for filter_type in self.root.findall(".//RangeFilter"):
            filter_id = self.safe_get_text(filter_type, 'id')
            if not filter_id:
                continue
                
            display_info = {
                'category': self.safe_get_text(filter_type, 'category'),
                'fields': {}
            }
            
            for field in filter_type.findall(".//AbstractField"):
                field_code = self.safe_get_text(field, 'code')
                if not field_code:
                    continue
                    
                field_type = field.get('type', 'unknown')
                field_info = {
                    'type': field_type,
                    'displayName': self.safe_get_text(field, 'displayName', field_code),
                }
                
                # Extract combo values if available
                combo_values = field.find('ComboValues')
                if combo_values is not None:
                    field_info['values'] = []
                    for val in combo_values.findall('ComboValue'):
                        code = self.safe_get_text(val, 'code')
                        display_name = self.safe_get_text(val, 'displayName', code)
                        if code:
                            field_info['values'].append({
                                'code': code,
                                'displayName': display_name
                            })
                    
                display_info['fields'][field_code] = field_info
                
            if display_info['fields']:  # Only add if we found some fields
                self.available_filters[filter_id] = display_info
            
    def get_example_value(self, field_type: str, field_code: str) -> str:
        """Generate appropriate example value based on field type."""
        if 'DoubleField' in field_type:
            if 'price' in field_code.lower():
                return '25.50'
            elif 'perc' in field_code.lower() or 'percent' in field_code.lower():
                return '5.5'
            else:
                return '100.0'
        elif 'IntField' in field_type:
            if 'volume' in field_code.lower():
                return '1000000'
            else:
                return '100'
        elif 'DateField' in field_type:
            return '20240101'  # YYYYMMDD format
        elif 'ComboField' in field_type:
            return 'AAA'  # For ratings, etc.
        else:
            return '10'  # Default fallback
            
    def generate_markdown(self) -> str:
        """Generate markdown documentation."""
        doc = []
        
        # Header
        doc.append("# IB Scanner API Documentation\n")
        
        # ScannerSubscription Usage Example
        doc.append("## Basic Usage Example\n")
        doc.append("```python")
        doc.append("from ib_insync import ScannerSubscription, TagValue\n")
        doc.append("# Create a scanner subscription")
        doc.append("sub = ScannerSubscription(")
        doc.append("    instrument='STK',")
        doc.append("    locationCode='STK.US.MAJOR',")
        doc.append("    scanCode='TOP_PERC_GAIN'")
        doc.append(")")
        doc.append("\n# Add filter criteria using TagValue objects")
        doc.append("tagValues = [")
        doc.append("    # Double values must be passed as strings")
        doc.append("    TagValue('priceAbove', '25.50'),      # scanner.filter.DoubleField")
        doc.append("    TagValue('changePercAbove', '5.5'),   # scanner.filter.DoubleField")
        doc.append("    TagValue('volumeAbove', '1000000'),   # scanner.filter.IntField")
        doc.append("    TagValue('marketCapAbove', '1e6'),    # scanner.filter.DoubleField")
        doc.append("    TagValue('moodyRatingAbove', 'AAA')   # scanner.filter.ComboField")
        doc.append("]")
        doc.append("\n# Request scanner data")
        doc.append("scanData = ib.reqScannerData(sub, [], tagValues)")
        doc.append("```\n")
        
        # Data Type Notes
        doc.append("## Important Notes About Data Types\n")
        doc.append("When using TagValue filters, all values must be passed as strings, but they should conform to the expected format:\n")
        doc.append("- `scanner.filter.DoubleField`: Pass numbers as strings (e.g., '25.50', '100.0')")
        doc.append("- `scanner.filter.IntField`: Pass integers as strings (e.g., '1000000', '100')")
        doc.append("- `scanner.filter.DateField`: Pass dates as 'YYYYMMDD' strings (e.g., '20240101')")
        doc.append("- `scanner.filter.ComboField`: Pass the exact code as string (e.g., 'AAA' for ratings)\n")
        
        # Available Filters
        if self.available_filters:
            doc.append("\n## Available TagValue Filters\n")
            for filter_id, info in sorted(self.available_filters.items()):
                doc.append(f"\n### {filter_id}\n")
                doc.append(f"Category: {info['category']}\n")
                
                doc.append("| Filter Code | Field Type | Description | Example Usage |")
                doc.append("|-------------|------------|-------------|---------------|")
                
                for field_code, field_info in info['fields'].items():
                    field_type = field_info['type']
                    example_value = self.get_example_value(field_type, field_code)
                    example = f"`TagValue('{field_code}', '{example_value}')`"
                    type_note = field_type.split('.')[-1]  # Get just the end part of the type
                    
                    doc.append(f"| {field_code} | {type_note} | {field_info['displayName']} | {example} |")
                    
                    if 'values' in field_info and field_info['values']:
                        doc.append("\nAllowed values for {}:".format(field_code))
                        for val in field_info['values']:
                            doc.append(f"- `{val['code']}`: {val['displayName']}")
                        doc.append("")
        
        # Location Codes
        if self.location_codes:
            doc.append("\n## Available Location Codes\n")
            doc.append("Use these codes with the `locationCode` parameter:\n")
            doc.append("| Location Code | Usage Example |")
            doc.append("|--------------|----------------|")
            for location in sorted(self.location_codes):
                doc.append(f"| {location} | `locationCode='{location}'` |")
            
        return "\n".join(doc)

def create_md_documentaion_from_scan_params_xml(file_path: str = 'ib_scan_params.xml'):
    # Create documentation generator
    generator = ScannerDocsGenerator(file_path)
    
    # Extract all necessary information
    generator.extract_scan_types()
    generator.extract_locations()
    generator.extract_instruments()
    generator.extract_filters()
    
    # Generate markdown documentation
    markdown_content = generator.generate_markdown()
    
    # Save to file
    with open('ib_scanner_documentation.md', 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print("Documentation has been saved to ib_scanner_documentation.md")
        
