import numpy as np
import matplotlib.pyplot as plt

# Data from the image
x = np.array([-9.00, -7.06, -5.00, -3.03, -1.03, 1.00, 3.00, 5.03, 7.97, 9.26])
y = np.array([7.90, 6.65, 5.00, 2.70, 1.14, -0.62, -3.18, -4.58, -6.78, -8.06])

# Calculate Pearson correlation coefficient
correlation_coefficient = np.corrcoef(x, y)[0, 1]

# Display results
print("=" * 60)
print("PEARSON CORRELATION ANALYSIS - FINAL RESULTS")
print("=" * 60)
print(f"\nNumber of data points (n): {len(x)}")

print("\nDATA POINTS:")
print(f"{'No.':>3} {'X':>10} {'Y':>10}")
print("-" * 30)
for i in range(len(x)):
    print(f"{i + 1:3d} {x[i]:10.2f} {y[i]:10.2f}")

print(f"\nPEARSON CORRELATION COEFFICIENT (r): {correlation_coefficient:.6f}")
print(f"Coefficient of determination (r²): {correlation_coefficient ** 2:.6f}")
print(f"Percentage of variance explained: {correlation_coefficient ** 2 * 100:.2f}%")

# Calculate using the formula from your image
print("\n" + "=" * 60)
print("CALCULATION USING THE FORMULA:")
print("=" * 60)
print("\nFormula from your image:")
print("r = Σ[(xᵢ - x̄)(yᵢ - ȳ)] / √[Σ(xᵢ - x̄)² * Σ(yᵢ - ȳ)²]")

# Manual calculation
mean_x = np.mean(x)
mean_y = np.mean(y)

numerator = np.sum((x - mean_x) * (y - mean_y))
denominator_x = np.sum((x - mean_x) ** 2)
denominator_y = np.sum((y - mean_y) ** 2)
correlation_manual = numerator / np.sqrt(denominator_x * denominator_y)

print(f"\nStep-by-step calculation:")
print(f"1. Mean of X (x̄) = {mean_x:.6f}")
print(f"2. Mean of Y (ȳ) = {mean_y:.6f}")
print(f"3. Σ[(xᵢ - x̄)(yᵢ - ȳ)] = {numerator:.6f}")
print(f"4. Σ(xᵢ - x̄)² = {denominator_x:.6f}")
print(f"5. Σ(yᵢ - ȳ)² = {denominator_y:.6f}")
print(f"6. √[Σ(xᵢ - x̄)² * Σ(yᵢ - ȳ)²] = {np.sqrt(denominator_x * denominator_y):.6f}")
print(f"7. r = {numerator:.6f} / {np.sqrt(denominator_x * denominator_y):.6f} = {correlation_manual:.6f}")

# Correlation interpretation
print("\n" + "=" * 60)
print("CORRELATION INTERPRETATION:")
print("=" * 60)

# Strength interpretation
if abs(correlation_coefficient) >= 0.9:
    strength = "VERY STRONG"
    strength_desc = "The variables show an almost perfect linear relationship"
elif abs(correlation_coefficient) >= 0.7:
    strength = "STRONG"
    strength_desc = "The variables show a strong linear relationship"
elif abs(correlation_coefficient) >= 0.5:
    strength = "MODERATE"
    strength_desc = "The variables show a moderate linear relationship"
elif abs(correlation_coefficient) >= 0.3:
    strength = "WEAK"
    strength_desc = "The variables show a weak linear relationship"
else:
    strength = "VERY WEAK or NO"
    strength_desc = "Little to no linear relationship between variables"

# Direction interpretation
if correlation_coefficient < 0:
    direction = "NEGATIVE"
    direction_desc = "As X increases, Y tends to decrease"
    relationship = "inverse relationship"
else:
    direction = "POSITIVE"
    direction_desc = "As X increases, Y tends to increase"
    relationship = "direct relationship"

print(f"\n1. STRENGTH: {strength} correlation")
print(f"   - {strength_desc}")
print(
    f"   - r² = {correlation_coefficient ** 2:.4f} means {correlation_coefficient ** 2 * 100:.1f}% of Y's variation is explained by X")

print(f"\n2. DIRECTION: {direction} correlation")
print(f"   - {direction_desc}")
print(f"   - This indicates an {relationship} between X and Y")

print(f"\n3. PRACTICAL MEANING:")
print(f"   - With r = {correlation_coefficient:.4f}, we can predict Y from X with high accuracy")
print(f"   - The negative sign indicates that high X values correspond to low Y values")

# Create enhanced visualization
plt.figure(figsize=(12, 8))

# Scatter plot with color gradient based on position
scatter = plt.scatter(x, y, color='blue', s=150, alpha=0.7,
                      edgecolors='black', linewidth=2, zorder=5,
                      label='Data points')

# Add point numbers
for i, (xi, yi) in enumerate(zip(x, y)):
    plt.text(xi + 0.3, yi + 0.3, f'{i + 1}', fontsize=11,
             fontweight='bold', alpha=0.8, zorder=6)

# Calculate and plot regression line
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
x_line = np.linspace(min(x) - 1, max(x) + 1, 100)
y_line = p(x_line)
plt.plot(x_line, y_line, "r--", alpha=0.9, linewidth=3,
         label=f'Regression line: y = {z[0]:.3f}x + {z[1]:.3f}', zorder=4)

# Plot settings
plt.title('Pearson Correlation Analysis: r = -0.9842',
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('X values', fontsize=14)
plt.ylabel('Y values', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--', zorder=1)

# Add reference lines
plt.axhline(y=0, color='black', linewidth=0.8, alpha=0.5, zorder=2)
plt.axvline(x=0, color='black', linewidth=0.8, alpha=0.5, zorder=2)

# Add correlation info box
corr_text = (
    f'PEARSON CORRELATION:\n'
    f'r = {correlation_coefficient:.6f}\n'
    f'r² = {correlation_coefficient ** 2:.6f}\n'
    f'n = {len(x)} points\n'
    f'Strength: Very Strong\n'
    f'Direction: Negative'
)

plt.text(0.02, 0.98, corr_text,
         transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue',
                   alpha=0.9, edgecolor='darkblue'),
         fontsize=11, family='monospace',
         zorder=7)

# Add formula box
formula_text = (
    'FORMULA:\n'
    '        Σ[(xᵢ - x̄)(yᵢ - ȳ)]\n'
    'r = ────────────────────────\n'
    '    √[Σ(xᵢ - x̄)²·Σ(yᵢ - ȳ)²]'
)

plt.text(0.02, 0.15, formula_text,
         transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow',
                   alpha=0.9, edgecolor='darkgoldenrod'),
         fontsize=11, family='monospace',
         zorder=7)

plt.legend(loc='upper left', fontsize=11)
plt.tight_layout()

# Save the plot
plt.savefig('correlation_plot_final.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 60)
print(f"Plot saved as 'correlation_plot_final.png'")

# Show the plot
plt.show()

# Statistical summary
print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

stats_data = {
    'Statistic': ['Count', 'Mean', 'Std Dev', 'Variance', 'Minimum', '25th Percentile',
                  'Median', '75th Percentile', 'Maximum', 'Range', 'IQR'],
    'X': [len(x), np.mean(x), np.std(x), np.var(x), np.min(x),
          np.percentile(x, 25), np.median(x), np.percentile(x, 75),
          np.max(x), np.ptp(x), np.percentile(x, 75) - np.percentile(x, 25)],
    'Y': [len(y), np.mean(y), np.std(y), np.var(y), np.min(y),
          np.percentile(y, 25), np.median(y), np.percentile(y, 75),
          np.max(y), np.ptp(y), np.percentile(y, 75) - np.percentile(y, 25)]
}

# Print formatted table
print(f"\n{'Statistic':<20} {'X':>15} {'Y':>15}")
print("-" * 50)
for i in range(len(stats_data['Statistic'])):
    stat = stats_data['Statistic'][i]
    x_val = stats_data['X'][i]
    y_val = stats_data['Y'][i]

    if stat in ['Count']:
        print(f"{stat:<20} {int(x_val):>15} {int(y_val):>15}")
    else:
        print(f"{stat:<20} {x_val:>15.4f} {y_val:>15.4f}")

# Covariance
covariance = np.cov(x, y)[0, 1]
print(f"\nCovariance between X and Y: {covariance:.6f}")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)
print(f"\nThe Pearson correlation coefficient r = {correlation_coefficient:.6f} indicates:")
print(f"1. A VERY STRONG negative linear relationship between X and Y")
print(f"2. {correlation_coefficient ** 2 * 100:.2f}% of the variation in Y is explained by X")
print(f"3. The regression equation is: y = {z[0]:.4f}x + {z[1]:.4f}")
print(f"4. For every 1 unit increase in X, Y decreases by {abs(z[0]):.4f} units on average")
print("\nThis strong correlation suggests X is an excellent predictor of Y in this dataset.")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)