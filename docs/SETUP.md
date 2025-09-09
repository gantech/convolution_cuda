# Setting Up GitHub Pages

To set up GitHub Pages for your CUDA convolution documentation, follow these steps:

1. **Push your changes to GitHub**:
   ```bash
   git add docs/
   git commit -m "Add documentation for GitHub Pages"
   git push origin main
   ```

2. **Enable GitHub Pages in your repository**:
   - Go to your repository on GitHub
   - Click on "Settings"
   - Scroll down to the "GitHub Pages" section
   - Under "Source", select the "main" branch and "/docs" folder
   - Click "Save"

3. **Wait for deployment**:
   GitHub will now build and deploy your site. This usually takes a few minutes.

4. **Access your documentation**:
   Your site will be available at `https://gantech.github.io/convolution_cuda/`

## Customizing Your Documentation

### Modifying the Theme

The documentation uses the Cayman theme. You can customize it by editing the `_config.yml` file.

### Adding New Implementation Pages

To add documentation for a new optimization technique:

1. Create a new Markdown file in the `docs/implementations/` directory
2. Follow the same format as the existing implementation pages
3. Add a link to the new page in `docs/index.md`

### Updating Performance Data

To update the performance chart with your actual benchmark results:

1. Edit the `docs/js/performance-chart.js` file
2. Update the data array with your measured performance numbers

### Adding Images

1. Place your images in the `docs/assets/` directory
2. Reference them in your Markdown files using relative paths:
   ```markdown
   ![Description](../assets/your_image.png)
   ```

## Best Practices for Documentation

1. **Keep it visual**: Add diagrams, charts, and memory access pattern visualizations
2. **Show code snippets**: Include the most relevant parts of your implementation
3. **Explain trade-offs**: Discuss the advantages and limitations of each approach
4. **Include benchmarks**: Show how performance improves with each optimization
5. **Link related sections**: Make it easy for readers to navigate between connected topics
