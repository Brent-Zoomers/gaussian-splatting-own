import torch

def cartesian_to_spherical(x, y, z):
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.atan2(y, x)
    phi = torch.acos(z / r)
    return r, theta, phi

def spherical_to_cartesian(r, theta, phi):
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    return x, y, z

# Test code
def test_conversion():
    # Generate some random Cartesian coordinates
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([2.0, 3.0])
    z = torch.tensor([3.0, 4.0])
    
    # Convert Cartesian to spherical
    r, theta, phi = cartesian_to_spherical(x, y, z)
    
    # Convert spherical back to Cartesian
    x_back, y_back, z_back = spherical_to_cartesian(r, theta, phi)
    
    # Print results
    print("Original Cartesian coordinates:")
    print("x:", x)
    print("y:", y)
    print("z:", z)
    print("\nConverted spherical coordinates:")
    print("r:", r)
    print("theta:", theta)
    print("phi:", phi)
    print("\nConverted back Cartesian coordinates:")
    print("x_back:", x_back)
    print("y_back:", y_back)
    print("z_back:", z_back)
    
    # Calculate difference (should be very small if conversions are accurate)
    diff_x = torch.abs(x - x_back)
    diff_y = torch.abs(y - y_back)
    diff_z = torch.abs(z - z_back)
    print("\nDifference (should be close to zero):")
    print("diff_x:", diff_x)
    print("diff_y:", diff_y)
    print("diff_z:", diff_z)

# Run the test
if __name__ == "__main__":
    test_conversion()
