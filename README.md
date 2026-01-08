
<h1>Thermal Colormap Conversion Pipeline</h1>

<p>
This Python script converts high-contrast thermal images with rainbow colormaps into
<em>white-hot thermograms</em>. Black/background regions are preserved, and output images
maintain the same directory structure as the input.
</p>

<hr>

<h2>Features</h2>
<ul>
  <li>Converts rainbow or false-color thermal images to white-hot grayscale</li>
  <li>Preserves black/background pixels</li>
  <li>Optional Gaussian blur for smoother output</li>
  <li>Recursive batch processing with folder structure preservation</li>
  <li>Progress tracking via <code>tqdm</code></li>
</ul>


<h2>Pipeline Overview</h2>

<ol>
  <li><strong>Load Image:</strong> OpenCV reads BGR thermal images.</li>
  <li><strong>Black Pixel Mask:</strong> Detect and preserve near-black/background pixels.</li>
  <li><strong>HSV Conversion:</strong> Convert BGR → HSV and extract hue channel.</li>
  <li><strong>HSV Shifting:</strong> Shift degree of color space to align with thermography standards .</li>
  <img src="https://github.com/SyedMuhammadZarif/Thermogram-Conversion-Pipeline-Rainbow-High-Contrast-to-White-Hot-/blob/69b74a7c457cbe3cafb3c0f284a6873df2d660bd/colorwheel.png" width="300">
  <li><strong>Hue → White-Hot Mapping:</strong> Apply Gaussian-based mapping to convert color-coded thermal values to intensity values (white = hottest, black = coldest).</li>
  <img src="https://github.com/SyedMuhammadZarif/Thermogram_Conversion_Pipeline/blob/e18aad6bba983eb93b9bbc893e202b0a70914902/pipeline.png" width="300">
  <li><strong>Post-Processing:</strong> Restore black pixels, optionally blur for smooth gradients.</li>
  <li><strong>Save Output:</strong> Preserve filenames and directory hierarchy.</li>
</ol>

<hr>

<h2>Configuration</h2>

<pre>
input_folder  = "STEP 1 Data Processing\\IEEE\\IEE_RESIZED"
output_folder = "STEP 1 Data Processing\\IEEE\\IEEE_Grayscaled"
</pre>

<hr>

<h2>Supported Formats</h2>

<ul>
  <li>PNG</li>
  <li>JPG / JPEG</li>
  <li>BMP</li>
  <li>TIFF</li>
</ul>

<hr>

<h2>Usage</h2>

<pre>
python thermal_colormap_to_whitehot.py
</pre>

<p>
Progress is displayed via <code>tqdm</code>. All outputs are saved to the specified folder.
</p>

<hr>

<h2>Output</h2>
<ul>
  <li>White-hot grayscale thermograms</li>
  <li>Preserved black/background regions</li>
  <li>Folder structure matches input</li>
</ul>

<hr>

<h2>License</h2>
<p>
Free for research and educational purposes. Modify for personal or lab pipelines.
</p>
