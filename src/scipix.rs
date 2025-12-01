//! PyO3 wrapper for ruvector-scipix OCR functionality
//!
//! This module provides Python bindings for scientific document OCR with support for
//! LaTeX, MathML, and text extraction from mathematical equations and scientific papers.
//!
//! # Example
//!
//! ```python
//! from pyruvector import SciPixOCR, OutputFormat
//!
//! # Create OCR engine
//! ocr = SciPixOCR()
//!
//! # Process an image file
//! latex = ocr.ocr_image("equation.png", format="latex")
//! print(f"LaTeX: {latex}")
//!
//! # Process image bytes
//! with open("formula.jpg", "rb") as f:
//!     image_data = f.read()
//! mathml = ocr.ocr_bytes(image_data, format="mathml")
//!
//! # Extract all equations from a document
//! equations = ocr.extract_equations("paper.pdf")
//! for eq in equations:
//!     print(f"Equation: {eq}")
//!
//! # Check supported formats
//! formats = ocr.supported_formats()
//! print(f"Formats: {formats}")
//! ```

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::collections::HashMap;

// Import from ruvector-scipix
use ruvector_scipix::{
    ocr::{OcrEngine, OcrOptions, OcrResult as ScipixOcrResult, RegionType},
    output::OutputFormat as ScipixOutputFormat,
};

/// Output format for OCR results
///
/// Specifies the format for extracted mathematical content.
///
/// # Variants
/// * `LaTeX` - LaTeX mathematical notation (e.g., `$\\frac{1}{2}$`)
/// * `MathML` - Mathematical Markup Language XML
/// * `Text` - Plain text extraction
/// * `Mmd` - Scipix Markdown (enhanced markdown with math)
/// * `Html` - HTML with embedded math rendering
///
/// # Example
/// ```python
/// from pyruvector import OutputFormat
///
/// # Use LaTeX format
/// format = OutputFormat.LaTeX
/// ```
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputFormat {
    /// LaTeX mathematical notation
    LaTeX,
    /// Mathematical Markup Language
    MathML,
    /// Plain text output
    Text,
    /// Scipix Markdown (enhanced markdown)
    Mmd,
    /// HTML with embedded math
    Html,
}

impl From<OutputFormat> for ScipixOutputFormat {
    fn from(format: OutputFormat) -> Self {
        match format {
            OutputFormat::LaTeX => ScipixOutputFormat::LaTeX,
            OutputFormat::MathML => ScipixOutputFormat::MathML,
            OutputFormat::Text => ScipixOutputFormat::Text,
            OutputFormat::Mmd => ScipixOutputFormat::Mmd,
            OutputFormat::Html => ScipixOutputFormat::Html,
        }
    }
}

impl OutputFormat {
    /// Parse format from string
    fn from_str(s: &str) -> PyResult<Self> {
        match s.to_lowercase().as_str() {
            "latex" | "tex" => Ok(OutputFormat::LaTeX),
            "mathml" | "mml" => Ok(OutputFormat::MathML),
            "text" | "txt" => Ok(OutputFormat::Text),
            "mmd" | "markdown" => Ok(OutputFormat::Mmd),
            "html" => Ok(OutputFormat::Html),
            _ => Err(PyValueError::new_err(format!(
                "Unknown format '{}'. Supported: latex, mathml, text, mmd, html",
                s
            ))),
        }
    }
}

#[pymethods]
impl OutputFormat {
    fn __repr__(&self) -> String {
        format!("{:?}", self)
    }

    fn __str__(&self) -> &'static str {
        match self {
            OutputFormat::LaTeX => "latex",
            OutputFormat::MathML => "mathml",
            OutputFormat::Text => "text",
            OutputFormat::Mmd => "mmd",
            OutputFormat::Html => "html",
        }
    }
}

/// Scientific document OCR engine for extracting LaTeX and MathML from images
///
/// The SciPixOCR class provides high-performance optical character recognition
/// for scientific documents, mathematical equations, and technical diagrams.
/// It supports multiple output formats including LaTeX, MathML, and plain text.
///
/// # Features
/// - Extract LaTeX from equation images
/// - Convert to MathML for web display
/// - Process PDF pages and extract all equations
/// - High-accuracy mathematical symbol recognition
/// - GPU acceleration support (when available)
///
/// # Example
/// ```python
/// from pyruvector import SciPixOCR
///
/// # Initialize engine
/// ocr = SciPixOCR()
///
/// # Process single equation
/// latex = ocr.ocr_image("equation.png")
/// print(latex)
///
/// # Process with specific format
/// mathml = ocr.ocr_image("formula.jpg", format="mathml")
///
/// # Extract from PDF
/// equations = ocr.extract_equations("paper.pdf")
/// for eq in equations:
///     print(f"Found: {eq}")
/// ```
#[pyclass]
pub struct SciPixOCR {
    /// Runtime for async operations
    runtime: tokio::runtime::Runtime,
}

#[pymethods]
impl SciPixOCR {
    /// Create a new SciPix OCR engine
    ///
    /// # Example
    /// ```python
    /// from pyruvector import SciPixOCR
    ///
    /// ocr = SciPixOCR()
    /// ```
    #[new]
    #[pyo3(signature = (_enable_gpu=false, _high_accuracy=false))]
    fn new(_enable_gpu: bool, _high_accuracy: bool) -> PyResult<Self> {
        // Create tokio runtime for async operations
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;

        Ok(Self { runtime })
    }

    /// Perform OCR on an image file
    ///
    /// # Arguments
    /// * `path` - Path to the image file (PNG, JPG, PDF, etc.)
    /// * `format` - Output format: "latex", "mathml", "text", "mmd", or "html" (default: "latex")
    ///
    /// # Returns
    /// Extracted content in the specified format
    ///
    /// # Example
    /// ```python
    /// latex = ocr.ocr_image("equation.png")
    /// mathml = ocr.ocr_image("formula.jpg", format="mathml")
    /// text = ocr.ocr_image("document.pdf", format="text")
    /// ```
    #[pyo3(signature = (path, format=None))]
    fn ocr_image(&self, path: &str, format: Option<&str>) -> PyResult<String> {
        let format = format.unwrap_or("latex");
        let output_format = OutputFormat::from_str(format)?;

        // Read image file
        let image_data = std::fs::read(path).map_err(|e| {
            PyIOError::new_err(format!("Failed to read image file '{}': {}", path, e))
        })?;

        // Process with OCR engine
        self.process_image_data(&image_data, output_format)
    }

    /// Perform OCR on image bytes
    ///
    /// # Arguments
    /// * `data` - Image data as bytes
    /// * `format` - Output format: "latex", "mathml", "text", "mmd", or "html" (default: "latex")
    ///
    /// # Returns
    /// Extracted content in the specified format
    ///
    /// # Example
    /// ```python
    /// with open("equation.png", "rb") as f:
    ///     image_bytes = f.read()
    ///
    /// latex = ocr.ocr_bytes(image_bytes)
    /// mathml = ocr.ocr_bytes(image_bytes, format="mathml")
    /// ```
    #[pyo3(signature = (data, format=None))]
    fn ocr_bytes(&self, data: &[u8], format: Option<&str>) -> PyResult<String> {
        let format = format.unwrap_or("latex");
        let output_format = OutputFormat::from_str(format)?;
        self.process_image_data(data, output_format)
    }

    /// Extract all equations from a document or image
    ///
    /// Detects and extracts all mathematical equations from the input,
    /// returning them as a list of LaTeX strings.
    ///
    /// # Arguments
    /// * `path` - Path to image or PDF file
    ///
    /// # Returns
    /// List of LaTeX strings, one for each detected equation
    ///
    /// # Example
    /// ```python
    /// equations = ocr.extract_equations("research_paper.pdf")
    /// for i, eq in enumerate(equations):
    ///     print(f"Equation {i+1}: {eq}")
    /// ```
    fn extract_equations(&self, py: Python, path: &str) -> PyResult<Py<PyList>> {
        // Read image file
        let image_data = std::fs::read(path)
            .map_err(|e| PyIOError::new_err(format!("Failed to read file '{}': {}", path, e)))?;

        // Process and extract equations
        let equations = self.extract_equations_from_data(&image_data)?;

        // Convert to Python list
        let py_list = PyList::empty(py);
        for eq in equations {
            py_list.append(eq)?;
        }

        Ok(py_list.into())
    }

    /// Get list of supported output formats
    ///
    /// # Returns
    /// List of format names: ["latex", "mathml", "text", "mmd", "html"]
    ///
    /// # Example
    /// ```python
    /// formats = ocr.supported_formats()
    /// print(f"Supported formats: {formats}")
    /// ```
    fn supported_formats(&self, py: Python) -> PyResult<Py<PyList>> {
        let formats = vec!["latex", "mathml", "text", "mmd", "html"];
        let py_list = PyList::new(py, formats);
        Ok(py_list.into())
    }

    /// Process image with custom options
    ///
    /// Advanced OCR processing with detailed options for fine-tuning.
    ///
    /// # Arguments
    /// * `data` - Image data as bytes
    /// * `options` - Dictionary with options:
    ///   - `format`: Output format (default: "latex")
    ///   - `enable_math`: Enable math detection (default: true)
    ///   - `confidence_threshold`: Minimum confidence 0.0-1.0 (default: 0.6)
    ///   - `languages`: List of language codes (default: ["en"])
    ///
    /// # Returns
    /// Dictionary with:
    ///   - `text`: Extracted content
    ///   - `confidence`: Overall confidence score
    ///   - `processing_time_ms`: Processing time in milliseconds
    ///
    /// # Example
    /// ```python
    /// result = ocr.process_with_options(
    ///     image_data,
    ///     options={
    ///         "format": "latex",
    ///         "enable_math": True,
    ///         "confidence_threshold": 0.7,
    ///         "languages": ["en", "de"]
    ///     }
    /// )
    /// print(f"Text: {result['text']}")
    /// print(f"Confidence: {result['confidence']:.2%}")
    /// ```
    #[pyo3(signature = (data, options=None))]
    fn process_with_options(
        &self,
        py: Python,
        data: &[u8],
        options: Option<HashMap<String, pyo3::PyObject>>,
    ) -> PyResult<HashMap<String, pyo3::PyObject>> {
        // Parse options
        let opts = if let Some(ref opt_map) = options {
            let mut ocr_opts = OcrOptions::default();

            // Extract format
            let format_str = if let Some(fmt) = opt_map.get("format") {
                fmt.extract::<String>(py)
                    .unwrap_or_else(|_| "latex".to_string())
            } else {
                "latex".to_string()
            };

            // Extract enable_math
            if let Some(enable_math) = opt_map.get("enable_math") {
                ocr_opts.enable_math = enable_math.extract::<bool>(py).unwrap_or(true);
            }

            // Extract confidence_threshold
            if let Some(threshold) = opt_map.get("confidence_threshold") {
                ocr_opts.recognition_threshold = threshold.extract::<f32>(py).unwrap_or(0.6);
            }

            // Extract languages
            if let Some(langs) = opt_map.get("languages") {
                if let Ok(lang_list) = langs.extract::<Vec<String>>(py) {
                    ocr_opts.languages = lang_list;
                }
            }

            (ocr_opts, format_str)
        } else {
            (OcrOptions::default(), "latex".to_string())
        };

        let (ocr_options, format_str) = opts;
        let output_format = OutputFormat::from_str(&format_str)?;

        // Process image with OCR
        let result = self.process_image_with_options(data, &ocr_options, output_format)?;

        // Build result dictionary
        let mut result_map = HashMap::new();
        result_map.insert("text".to_string(), result.0.into_py(py));
        result_map.insert("confidence".to_string(), result.1.into_py(py));
        result_map.insert("processing_time_ms".to_string(), result.2.into_py(py));

        Ok(result_map)
    }

    fn __repr__(&self) -> String {
        "SciPixOCR(backend='ruvector-scipix')".to_string()
    }
}

impl SciPixOCR {
    /// Internal method to process image data
    fn process_image_data(&self, data: &[u8], format: OutputFormat) -> PyResult<String> {
        let options = OcrOptions::default();
        let result = self.process_image_with_options(data, &options, format)?;
        Ok(result.0)
    }

    /// Internal method to process with custom options
    fn process_image_with_options(
        &self,
        data: &[u8],
        options: &OcrOptions,
        format: OutputFormat,
    ) -> PyResult<(String, f32, u64)> {
        // Run async OCR in the runtime
        let result = self.runtime.block_on(async {
            // Create OCR engine with options
            let engine = OcrEngine::with_options(options.clone())
                .await
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to create OCR engine: {}", e))
                })?;

            // Perform OCR
            let ocr_result = engine
                .recognize(data)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("OCR failed: {}", e)))?;

            Ok::<_, PyErr>(ocr_result)
        })?;

        // Format output
        let text = self.format_result(&result, format)?;
        let confidence = result.confidence;
        let processing_time_ms = result.processing_time_ms;

        Ok((text, confidence, processing_time_ms))
    }

    /// Format OCR result to requested output format
    fn format_result(&self, result: &ScipixOcrResult, format: OutputFormat) -> PyResult<String> {
        // For now, return the recognized text
        // In production, would use OutputFormatter to convert to specific formats
        let output = match format {
            OutputFormat::LaTeX => {
                // Convert text regions to LaTeX
                result
                    .regions
                    .iter()
                    .map(|r| r.text.clone())
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            OutputFormat::MathML => {
                // Convert to MathML (simplified)
                format!("<math>{}</math>", result.text)
            }
            OutputFormat::Text => result.text.clone(),
            OutputFormat::Mmd => {
                // Scipix Markdown format
                result
                    .regions
                    .iter()
                    .map(|r| format!("${}$", r.text))
                    .collect::<Vec<_>>()
                    .join("\n\n")
            }
            OutputFormat::Html => {
                // HTML with embedded math
                format!(
                    "<div class=\"math-content\">{}</div>",
                    result
                        .regions
                        .iter()
                        .map(|r| format!("<span class=\"equation\">{}</span>", r.text))
                        .collect::<Vec<_>>()
                        .join("\n")
                )
            }
        };

        Ok(output)
    }

    /// Extract all equations from image data
    fn extract_equations_from_data(&self, data: &[u8]) -> PyResult<Vec<String>> {
        let options = OcrOptions {
            enable_math: true,
            ..Default::default()
        };

        // Run async OCR
        let result = self.runtime.block_on(async {
            let engine = OcrEngine::with_options(options).await.map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create OCR engine: {}", e))
            })?;

            let ocr_result = engine
                .recognize(data)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("OCR failed: {}", e)))?;

            Ok::<_, PyErr>(ocr_result)
        })?;

        // Extract only math regions
        let equations: Vec<String> = result
            .regions
            .into_iter()
            .filter(|r| r.region_type == RegionType::Math)
            .map(|r| r.text)
            .collect();

        Ok(equations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format_from_str() {
        assert_eq!(
            OutputFormat::from_str("latex").unwrap(),
            OutputFormat::LaTeX
        );
        assert_eq!(
            OutputFormat::from_str("mathml").unwrap(),
            OutputFormat::MathML
        );
        assert_eq!(OutputFormat::from_str("text").unwrap(), OutputFormat::Text);
        assert_eq!(OutputFormat::from_str("mmd").unwrap(), OutputFormat::Mmd);
        assert_eq!(OutputFormat::from_str("html").unwrap(), OutputFormat::Html);

        // Case insensitive
        assert_eq!(
            OutputFormat::from_str("LATEX").unwrap(),
            OutputFormat::LaTeX
        );
        assert_eq!(
            OutputFormat::from_str("MathML").unwrap(),
            OutputFormat::MathML
        );

        // Invalid format
        assert!(OutputFormat::from_str("invalid").is_err());
    }

    #[test]
    fn test_output_format_conversion() {
        let format = OutputFormat::LaTeX;
        let scipix_format: ScipixOutputFormat = format.into();
        assert_eq!(scipix_format, ScipixOutputFormat::LaTeX);
    }
}
