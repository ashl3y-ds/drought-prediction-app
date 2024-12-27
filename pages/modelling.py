st.write("### Customized Confusion Matrix (Dynamic Color Based on Values):")
fig, ax = plt.subplots(figsize=(5, 5))  # Adjust figure size as needed

# Set custom background colors for the figure and axes
fig.patch.set_facecolor("#808080")  # Background color for the figure
ax.set_facecolor("#808080")         # Background color for the axes

# Number of classes
classes = np.unique(y)
n_classes = len(classes)

# Normalizing the values to create dynamic coloring (scaling between 0 and 1)
normalized_cm = cm / cm.max()

# Draw grid for the matrix
ax.set_xticks(np.arange(n_classes) + 0.5, minor=False)
ax.set_yticks(np.arange(n_classes) + 0.5, minor=False)
ax.set_xticks(np.arange(n_classes + 1), minor=True)
ax.set_yticks(np.arange(n_classes + 1), minor=True)

# Configure grid and labels
ax.set_xticklabels(classes, fontsize=12, fontweight="bold", color="black")
ax.set_yticklabels(classes, fontsize=12, fontweight="bold", color="black")
ax.xaxis.tick_top()  # Move the x-axis labels to the top
ax.xaxis.set_label_position('top')
ax.tick_params(axis='both', which='both', length=0)  # Hide tick marks

# Customize appearance
ax.grid(False)  # Turn off the default grid
ax.grid(True, which="minor", color="black", linewidth=1)  # Add gridlines around cells
ax.set_xlim(0, n_classes)
ax.set_ylim(n_classes, 0)

# Add dynamic background colors and text
for i in range(n_classes):
    for j in range(n_classes):
        # Compute a color intensity based on the value
        intensity = normalized_cm[i, j]
        color = plt.cm.Blues(intensity)  # Use Blues colormap to assign colors

        # Draw the cell rectangle with dynamic color
        rect = plt.Rectangle((j, i), 1, 1, facecolor=color, edgecolor="black")
        ax.add_patch(rect)

        # Add the value as text (always in black)
        ax.text(j + 0.5, i + 0.5, str(cm[i, j]), 
                ha="center", va="center", fontsize=12, color="black")

# Label titles
ax.set_xlabel("Predicted Labels", fontsize=12, fontweight="bold", color="black", labelpad=20)
ax.set_ylabel("True Labels", fontsize=12, fontweight="bold", color="black", labelpad=20)
ax.set_title("Customized Confusion Matrix", fontsize=14, fontweight="bold", color="black", pad=20)

# Tight layout
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# Display the plot
st.pyplot(fig)
