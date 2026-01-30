"""
LLM Service - Claude Haiku integration for data analysis insights
Fixed JSON parsing with proper sanitization
Shorter, bullet-point summaries
"""
import os
import json
import re
from anthropic import Anthropic
from typing import Any, Optional


def sanitize_for_json(text: str) -> str:
    """
    Sanitize text to be safe for JSON embedding.
    Removes or escapes control characters that break JSON parsing.
    """
    if not text:
        return ""
    
    # Replace common problematic characters
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Replace tabs with spaces
    text = text.replace('\t', '    ')
    
    # Normalize line endings to \n
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove other control characters (except newline which we handle separately)
    # Control chars are 0x00-0x1F except 0x0A (newline) and 0x0D (carriage return, already replaced)
    control_chars = ''.join(chr(i) for i in range(32) if i not in (10,))
    text = text.translate(str.maketrans('', '', control_chars))
    
    return text


def extract_json_from_response(response_text: str) -> dict:
    """
    Robustly extract JSON from LLM response, handling various formats.
    """
    text = response_text.strip()
    
    # Remove markdown code blocks
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    text = text.strip()
    
    # Try to find JSON object boundaries
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx + 1]
    
    # Sanitize the text
    text = sanitize_for_json(text)
    
    # Try parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try fixing common issues
        
        # Fix unescaped newlines in string values
        # This regex finds strings and escapes newlines within them
        def escape_newlines_in_strings(match):
            content = match.group(1)
            content = content.replace('\n', '\\n')
            return f'"{content}"'
        
        # Match strings (simplified - doesn't handle all edge cases but covers most)
        fixed_text = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', 
                          lambda m: '"' + m.group(1).replace('\n', '\\n') + '"', 
                          text)
        
        try:
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            pass
        
        # Last resort: try to extract key fields manually
        print(f"JSON parse failed, attempting manual extraction. Error: {e}")
        print(f"Response text (first 500 chars): {text[:500]}")
        
        # Return a minimal valid structure
        raise ValueError(f"Could not parse JSON from LLM response: {e}")


class LLMService:
    """Service for generating LLM-powered analysis insights"""
    
    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("WARNING: ANTHROPIC_API_KEY not set. LLM features will be limited.")
            self.client = None
        else:
            self.client = Anthropic(api_key=api_key)
        
        self.model = "claude-haiku-4-5-20251001"
    
    async def summarize_text_column(self, column_name: str, sample_values: list[str]) -> str:
        """Generate a short 2-3 sentence summary of what a text column contains"""
        
        if not self.client:
            return f"Text column containing varied textual data."
        
        # Sanitize sample values
        clean_samples = []
        for v in sample_values[:12]:
            clean_v = sanitize_for_json(str(v)[:150])
            clean_samples.append(f"- {clean_v}")
        sample_text = "\n".join(clean_samples)
        
        prompt = f"""Analyze this text column from a dataset and write a 2-3 sentence summary describing what kind of data it contains.

Column name: {column_name}

Sample values:
{sample_text}

Write a brief, informative summary (2-3 sentences) describing:
- What type of text data this appears to be
- The general content/theme
- Any patterns you notice

Respond with ONLY the summary text, no labels or prefixes. Do not use any special characters or formatting."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = response.content[0].text.strip()
            # Sanitize the response
            return sanitize_for_json(result)
            
        except Exception as e:
            print(f"Text summary error: {e}")
            return f"Text column '{column_name}' contains varied textual data."
    
    async def analyze(
        self, 
        dataset_info: dict, 
        columns: list[dict],
        schema_content: Optional[str] = None
    ) -> dict[str, Any]:
        """Generate comprehensive analysis using LLM"""
        
        if not self.client:
            return self._fallback_analysis(dataset_info, columns)
        
        # Build detailed column summary for prompt
        column_summaries = []
        numeric_cols = []
        categorical_cols = []
        text_cols = []
        flagged_cols = []
        
        for col in columns:
            summary = f"- **{col['name']}** ({col['dtype']}): "
            summary += f"{col['missing']} missing ({col['missing_pct']}%)"
            
            if col.get('quality_flags'):
                summary += f" [FLAGS: {', '.join(col['quality_flags'])}]"
                flagged_cols.append(col['name'])
            
            if col.get('stats'):
                stats = col['stats']
                if 'mean' in stats and stats['mean'] is not None:
                    summary += f"\n  Statistics: min={stats.get('min')}, max={stats.get('max')}, mean={stats.get('mean'):.2f}, median={stats.get('median')}, std={stats.get('std'):.2f}"
                    if stats.get('outliers_low', 0) > 0 or stats.get('outliers_high', 0) > 0:
                        summary += f", outliers: {stats.get('outliers_low', 0)} low, {stats.get('outliers_high', 0)} high"
                    numeric_cols.append(col['name'])
                elif 'unique_count' in stats:
                    summary += f", {stats['unique_count']} unique values"
                    if 'top_values' in stats and stats['top_values']:
                        top_3 = stats['top_values'][:3]
                        top_str = ", ".join([f"{v['value']}({v['percent']}%)" for v in top_3])
                        summary += f"\n  Top values: {top_str}"
                    if col['dtype'] == 'categorical':
                        categorical_cols.append(col['name'])
                    elif col['dtype'] == 'text':
                        text_cols.append(col['name'])
            
            column_summaries.append(summary)
        
        columns_text = "\n".join(column_summaries)
        
        # Dataset characteristics summary
        characteristics = f"""
DATASET CHARACTERISTICS:
- Total records: {dataset_info['rows']}
- Total columns: {dataset_info['columns']}
- Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}
- Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols[:10])}{'...' if len(categorical_cols) > 10 else ''}
- Text columns ({len(text_cols)}): {', '.join(text_cols[:5])}{'...' if len(text_cols) > 5 else ''}
- Columns with quality issues ({len(flagged_cols)}): {', '.join(flagged_cols[:10])}{'...' if len(flagged_cols) > 10 else ''}
"""
        
        schema_section = ""
        if schema_content:
            # Sanitize schema content
            clean_schema = sanitize_for_json(schema_content[:2000])
            schema_section = f"\n\nSCHEMA/DESCRIPTION PROVIDED BY USER:\n{clean_schema}\n"
        
        prompt = f"""You are an expert data analyst. Analyze this dataset and provide a CONCISE summary.

FILENAME: {dataset_info.get('filename', 'unknown')}
{characteristics}

DETAILED COLUMN INFORMATION:
{columns_text}
{schema_section}

Provide your analysis in JSON format with SHORT, POWERFUL bullet points:

{{
    "summary": [
        "Bullet 1: What this dataset is (one short sentence)",
        "Bullet 2: Key numeric insight with actual numbers",
        "Bullet 3: Key categorical insight",
        "Bullet 4: Data quality note if relevant",
        "Bullet 5: Most important pattern or finding"
    ],
    
    "insights": [
        "Column X: ranges 0-100, mean 45, high variance",
        "Column Y: 85% values are category A",
        "Strong correlation suspected between X and Z",
        "15% missing data in critical columns",
        "Outliers detected in numeric fields"
    ],
    
    "limitations": [
        "Missing values in X, Y columns need handling",
        "Cannot determine causation from this data",
        "Sample may not be representative"
    ]
}}

CRITICAL RULES:
- Summary: MAX 10 bullet points, each under 15 words
- Each bullet is a standalone fact, not a sentence fragment
- Use actual numbers from the data
- No fluff words, no "the data shows", just facts
- Insights: 5-7 specific findings with numbers
- Limitations: 2-3 concrete issues
- Output ONLY valid JSON, no markdown

Respond with JSON only."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            # Extract and parse JSON
            result = extract_json_from_response(response_text)
            
            # Sanitize all string values in result
            if 'summary' in result:
                if isinstance(result['summary'], list):
                    result['summary'] = [sanitize_for_json(str(s)) for s in result['summary']]
                else:
                    # If it's a string, convert to list
                    result['summary'] = [sanitize_for_json(str(result['summary']))]
            
            if 'insights' in result:
                result['insights'] = [sanitize_for_json(str(i)) for i in result['insights']]
            
            if 'limitations' in result:
                result['limitations'] = [sanitize_for_json(str(l)) for l in result['limitations']]
            
            return result
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return self._fallback_analysis(dataset_info, columns)
        except ValueError as e:
            print(f"Value error: {e}")
            return self._fallback_analysis(dataset_info, columns)
        except Exception as e:
            print(f"LLM error: {e}")
            return self._fallback_analysis(dataset_info, columns)
    
    async def suggest_plot(self, columns: list[dict]) -> dict[str, Any]:
        """Suggest appropriate plot type for given columns"""
        
        if not self.client:
            return self._fallback_plot_suggestion(columns)
        
        col_details = []
        for col in columns:
            detail = f"- {col['name']}: type={col['dtype']}"
            if col.get('stats'):
                if 'unique_count' in col['stats']:
                    detail += f", {col['stats']['unique_count']} unique values"
                if 'mean' in col['stats']:
                    detail += f", mean={col['stats']['mean']}, std={col['stats'].get('std', 'N/A')}"
            col_details.append(detail)
        
        num_cols = len(columns)
        
        if num_cols == 1:
            prompt = f"""The user selected 1 column from their dataset:
{chr(10).join(col_details)}

Suggest which additional columns would create an interesting visualization. Also recommend the best plot type.

Respond in JSON only, no markdown:
{{"plot_type": "histogram or boxplot or violin or bar", "x_column": "{columns[0]['name']}", "y_column": null, "reasoning": "Brief explanation"}}"""
        elif num_cols == 2:
            prompt = f"""The user selected 2 columns:
{chr(10).join(col_details)}

Suggest the best visualization for these two columns.

Respond in JSON only, no markdown:
{{"plot_type": "scatter or line or grouped_bar or heatmap_cat", "x_column": "first_column", "y_column": "second_column", "reasoning": "Brief explanation"}}"""
        else:
            prompt = f"""The user selected {num_cols} columns:
{chr(10).join(col_details)}

Suggest the best multi-variable visualization.

Respond in JSON only, no markdown:
{{"plot_type": "scatter_color or bubble or correlation or pairplot", "x_column": "column_for_x", "y_column": "column_for_y", "reasoning": "Brief explanation"}}"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            result = extract_json_from_response(response_text)
            
            # Sanitize reasoning
            if 'reasoning' in result:
                result['reasoning'] = sanitize_for_json(result['reasoning'])
            
            return result
            
        except Exception as e:
            print(f"Plot suggestion error: {e}")
            return self._fallback_plot_suggestion(columns)
    
    def _fallback_analysis(self, dataset_info: dict, columns: list[dict]) -> dict[str, Any]:
        """Fallback analysis when LLM is unavailable"""
        
        numeric_cols = [c for c in columns if c['dtype'].startswith('numeric')]
        cat_cols = [c for c in columns if c['dtype'] == 'categorical']
        text_cols = [c for c in columns if c['dtype'] == 'text']
        missing_cols = [c for c in columns if c['missing'] > 0]
        flagged_cols = [c for c in columns if c.get('quality_flags')]
        
        # Build concise bullet-point summary
        summary = [
            f"{dataset_info['rows']:,} records, {dataset_info['columns']} columns",
            f"{len(numeric_cols)} numeric, {len(cat_cols)} categorical, {len(text_cols)} text columns"
        ]
        
        if missing_cols:
            worst = max(missing_cols, key=lambda x: x['missing_pct'])
            summary.append(f"{len(missing_cols)} columns have missing values (worst: {worst['name']} at {worst['missing_pct']}%)")
        
        if numeric_cols and numeric_cols[0].get('stats'):
            col = numeric_cols[0]
            stats = col['stats']
            if 'mean' in stats:
                summary.append(f"{col['name']}: range {stats.get('min')}-{stats.get('max')}, mean {stats.get('mean'):.1f}")
        
        if cat_cols and cat_cols[0].get('stats', {}).get('top_values'):
            col = cat_cols[0]
            top = col['stats']['top_values'][0]
            summary.append(f"{col['name']}: {col['stats']['unique_count']} categories, top is '{top['value']}' ({top['percent']}%)")
        
        if flagged_cols:
            flags = set()
            for c in flagged_cols:
                flags.update(c.get('quality_flags', []))
            summary.append(f"Quality flags: {', '.join(flags)}")
        
        # Build meaningful insights
        insights = []
        
        if numeric_cols:
            col = numeric_cols[0]
            if col.get('stats') and 'mean' in col['stats']:
                stats = col['stats']
                outliers = stats.get('outliers_low', 0) + stats.get('outliers_high', 0)
                insights.append(f"{col['name']}: {stats.get('min')} to {stats.get('max')}, std={stats.get('std'):.2f}, {outliers} outliers")
        
        if cat_cols:
            insights.append(f"{len(cat_cols)} categorical columns for grouping/segmentation")
        
        if missing_cols:
            insights.append(f"{len(missing_cols)} columns need missing value treatment")
        
        for c in flagged_cols[:2]:
            if 'single_value' in c.get('quality_flags', []):
                insights.append(f"{c['name']}: single value only - remove from analysis")
            if 'potential_id' in c.get('quality_flags', []):
                insights.append(f"{c['name']}: appears to be ID column")
        
        insights.append(f"Dataset size: {'sufficient' if dataset_info['rows'] > 100 else 'limited'} for statistical analysis")
        
        limitations = [
            "Automated analysis - verify with domain expertise",
            "Correlation does not imply causation",
            "Validate findings on held-out data"
        ]
        
        return {
            "summary": summary[:10],  # Max 10 bullets
            "insights": insights[:7],
            "limitations": limitations
        }
    
    def _fallback_plot_suggestion(self, columns: list[dict]) -> dict[str, Any]:
        """Fallback plot suggestion when LLM is unavailable"""
        
        if len(columns) == 1:
            col = columns[0]
            if col['dtype'].startswith('numeric'):
                return {
                    "plot_type": "histogram",
                    "x_column": col['name'],
                    "y_column": None,
                    "reasoning": f"Histogram shows distribution of numeric column '{col['name']}'."
                }
            else:
                return {
                    "plot_type": "bar",
                    "x_column": col['name'],
                    "y_column": None,
                    "reasoning": f"Bar chart shows frequency of categorical column '{col['name']}'."
                }
        
        elif len(columns) == 2:
            numeric_cols = [c for c in columns if c['dtype'].startswith('numeric')]
            cat_cols = [c for c in columns if c['dtype'] == 'categorical']
            
            if len(numeric_cols) == 2:
                return {
                    "plot_type": "scatter",
                    "x_column": numeric_cols[0]['name'],
                    "y_column": numeric_cols[1]['name'],
                    "reasoning": f"Scatter plot shows relationship between two numeric variables."
                }
            elif len(numeric_cols) == 1 and len(cat_cols) == 1:
                return {
                    "plot_type": "grouped_bar",
                    "x_column": cat_cols[0]['name'],
                    "y_column": numeric_cols[0]['name'],
                    "reasoning": f"Grouped bar shows {numeric_cols[0]['name']} by {cat_cols[0]['name']}."
                }
            elif len(cat_cols) == 2:
                return {
                    "plot_type": "heatmap_cat",
                    "x_column": cat_cols[0]['name'],
                    "y_column": cat_cols[1]['name'],
                    "reasoning": f"Heatmap shows frequency across two categorical variables."
                }
        
        elif len(columns) >= 3:
            numeric_cols = [c for c in columns if c['dtype'].startswith('numeric')]
            
            if len(numeric_cols) >= 3:
                return {
                    "plot_type": "scatter_color",
                    "x_column": numeric_cols[0]['name'],
                    "y_column": numeric_cols[1]['name'],
                    "reasoning": f"Colored scatter shows 3 numeric variables simultaneously."
                }
            elif len(numeric_cols) >= 2:
                return {
                    "plot_type": "correlation",
                    "x_column": columns[0]['name'],
                    "y_column": columns[1]['name'],
                    "reasoning": "Correlation matrix shows all numeric relationships."
                }
        
        # Default fallback
        return {
            "plot_type": "histogram",
            "x_column": columns[0]['name'],
            "y_column": None,
            "reasoning": "Starting with histogram to explore distribution."
        }