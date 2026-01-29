"""
LLM Service - Claude Haiku integration for data analysis insights
Fixed JSON parsing with proper sanitization
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
        
        prompt = f"""You are an expert data analyst providing a comprehensive exploratory data analysis report. Analyze this dataset thoroughly:

FILENAME: {dataset_info.get('filename', 'unknown')}
{characteristics}

DETAILED COLUMN INFORMATION:
{columns_text}
{schema_section}

Provide your analysis in the following JSON format. Be COMPREHENSIVE and SPECIFIC:

{{
    "summary": "Write 2 full paragraphs of flowing prose. Paragraph 1: Describe what this dataset contains, what each record represents, key variables. Paragraph 2: Describe data quality, notable distributions, preprocessing needs. Use plain text only, no special formatting.",
    
    "insights": [
        "INSIGHT 1: Specific finding about data distribution with actual numbers",
        "INSIGHT 2: Finding about relationships or patterns between variables",
        "INSIGHT 3: Data quality observation with specific columns affected",
        "INSIGHT 4: Outlier or anomaly finding with numbers",
        "INSIGHT 5: Business or domain insight from the data",
        "INSIGHT 6: Recommendation for analysis approach",
        "INSIGHT 7: Notable statistic or surprising finding"
    ],
    
    "limitations": [
        "LIMITATION 1: Specific data quality issue and its impact",
        "LIMITATION 2: Coverage or sampling limitation",
        "LIMITATION 3: What questions this data cannot answer"
    ]
}}

CRITICAL FORMATTING RULES:
- Output ONLY valid JSON, no markdown, no code blocks
- Use plain ASCII text only - no special characters, no unicode
- Do not use actual newlines inside string values - write as continuous text
- Each insight and limitation should be a single line of text
- Keep text simple and avoid quotation marks within strings

Respond with the JSON object only."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            # Extract and parse JSON
            result = extract_json_from_response(response_text)
            
            # Sanitize all string values in result
            if 'summary' in result:
                result['summary'] = sanitize_for_json(result['summary'])
                # Add paragraph break for display
                result['summary'] = result['summary'].replace('Paragraph 2:', '\n\n')
            
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
        
        # Build comprehensive summary
        summary_p1 = f"This dataset contains {dataset_info['rows']:,} records across {dataset_info['columns']} columns. "
        summary_p1 += f"The data includes {len(numeric_cols)} numeric variables suitable for statistical analysis, "
        summary_p1 += f"{len(cat_cols)} categorical variables for grouping and segmentation"
        if text_cols:
            summary_p1 += f", and {len(text_cols)} text fields containing unstructured data"
        summary_p1 += ". Each record appears to represent a distinct observation or entity in the dataset."
        
        summary_p2 = f"Data quality assessment reveals that {len(missing_cols)} columns contain missing values that may require imputation or careful handling. "
        if flagged_cols:
            flags = []
            for c in flagged_cols:
                flags.extend(c.get('quality_flags', []))
            summary_p2 += f"Quality flags have been raised for {len(flagged_cols)} columns, including issues such as: {', '.join(set(flags))}. "
        summary_p2 += "The numeric variables show varying distributions that should be explored through visualization before modeling. "
        summary_p2 += "This dataset appears suitable for exploratory analysis, statistical testing, and potentially predictive modeling depending on the target variable."
        
        summary = f"{summary_p1}\n\n{summary_p2}"
        
        # Build meaningful insights
        insights = []
        
        if numeric_cols:
            col = numeric_cols[0]
            if col.get('stats') and 'mean' in col['stats']:
                stats = col['stats']
                insights.append(f"The column '{col['name']}' ranges from {stats.get('min')} to {stats.get('max')} with a mean of {stats.get('mean'):.2f} and standard deviation of {stats.get('std'):.2f}, indicating {'high' if stats.get('std', 0) > stats.get('mean', 1) else 'moderate'} variability in the data.")
        
        if cat_cols:
            insights.append(f"Found {len(cat_cols)} categorical columns ({', '.join([c['name'] for c in cat_cols[:3]])}) that can be used for grouping, stratification, or as features in classification models.")
        
        if missing_cols:
            worst = max(missing_cols, key=lambda x: x['missing_pct'])
            insights.append(f"Missing data is present in {len(missing_cols)} columns. The most affected is '{worst['name']}' with {worst['missing_pct']}% missing values - consider imputation strategies or exclusion depending on analysis goals.")
        
        if flagged_cols:
            for c in flagged_cols[:2]:
                if 'single_value' in c.get('quality_flags', []):
                    insights.append(f"Column '{c['name']}' contains only a single unique value and provides no discriminative information - recommend removing from analysis.")
                if 'potential_id' in c.get('quality_flags', []):
                    insights.append(f"Column '{c['name']}' appears to be an identifier column with high cardinality - should be excluded from statistical modeling but may be useful for record tracking.")
        
        insights.append(f"Dataset has {dataset_info['rows']:,} observations which is {'sufficient' if dataset_info['rows'] > 100 else 'limited'} for most statistical analyses and machine learning approaches.")
        
        # Pad to minimum 5 insights
        while len(insights) < 5:
            insights.append("Further domain-specific analysis recommended to uncover deeper patterns and relationships in the data.")
        
        limitations = [
            f"Automated analysis may miss domain-specific patterns - recommend consulting with subject matter experts familiar with this type of data.",
            f"Statistical relationships identified do not imply causation - experimental design or causal inference methods needed for causal claims.",
            f"Results should be validated on held-out data before deployment, and any derived insights verified against external sources when possible."
        ]
        
        return {
            "summary": summary,
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
                    "reasoning": f"For the single numeric column '{col['name']}', a histogram shows the distribution. Consider adding another numeric column to create a scatter plot, or a categorical column for grouped comparisons."
                }
            else:
                return {
                    "plot_type": "bar",
                    "x_column": col['name'],
                    "y_column": None,
                    "reasoning": f"For the categorical column '{col['name']}', a bar chart shows frequency counts. Consider adding a numeric column to see averages by category."
                }
        
        elif len(columns) == 2:
            numeric_cols = [c for c in columns if c['dtype'].startswith('numeric')]
            cat_cols = [c for c in columns if c['dtype'] == 'categorical']
            
            if len(numeric_cols) == 2:
                return {
                    "plot_type": "scatter",
                    "x_column": numeric_cols[0]['name'],
                    "y_column": numeric_cols[1]['name'],
                    "reasoning": f"A scatter plot reveals the relationship between '{numeric_cols[0]['name']}' and '{numeric_cols[1]['name']}'. Add a third categorical column for color-coding by groups."
                }
            elif len(numeric_cols) == 1 and len(cat_cols) == 1:
                return {
                    "plot_type": "grouped_bar",
                    "x_column": cat_cols[0]['name'],
                    "y_column": numeric_cols[0]['name'],
                    "reasoning": f"A grouped bar chart shows average '{numeric_cols[0]['name']}' across categories of '{cat_cols[0]['name']}'."
                }
            elif len(cat_cols) == 2:
                return {
                    "plot_type": "heatmap_cat",
                    "x_column": cat_cols[0]['name'],
                    "y_column": cat_cols[1]['name'],
                    "reasoning": f"A heatmap shows the frequency distribution across both categorical variables '{cat_cols[0]['name']}' and '{cat_cols[1]['name']}'."
                }
        
        elif len(columns) >= 3:
            numeric_cols = [c for c in columns if c['dtype'].startswith('numeric')]
            
            if len(numeric_cols) >= 3:
                return {
                    "plot_type": "scatter_color",
                    "x_column": numeric_cols[0]['name'],
                    "y_column": numeric_cols[1]['name'],
                    "reasoning": f"With 3+ variables, a colored scatter plot shows '{numeric_cols[0]['name']}' vs '{numeric_cols[1]['name']}' with color representing '{numeric_cols[2]['name']}'. For an overview of all relationships, try the correlation matrix."
                }
            elif len(numeric_cols) >= 2:
                return {
                    "plot_type": "correlation",
                    "x_column": columns[0]['name'],
                    "y_column": columns[1]['name'],
                    "reasoning": "A correlation matrix shows relationships between all numeric variables at once."
                }
        
        # Default fallback
        return {
            "plot_type": "histogram",
            "x_column": columns[0]['name'],
            "y_column": None,
            "reasoning": "Starting with a histogram to explore the distribution of the first selected variable."
        }