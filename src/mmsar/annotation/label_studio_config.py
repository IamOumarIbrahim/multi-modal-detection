"""Programmatic generation and validation of Label Studio annotation configurations."""

import xml.etree.ElementTree as ET


DEFAULT_LABEL_NAME = "Person_Detected"


def generate_label_studio_config(label_name: str = DEFAULT_LABEL_NAME) -> str:
    """Generate Label Studio XML configuration for object detection bounding boxes.

    Args:
        label_name: Name of the object detection class label.

    Returns:
        Formatted XML string for Label Studio project configuration.
    """
    return (
        f"<View>\n"
        f'  <Image name="image" value="$image"/>\n'
        f'  <RectangleLabels name="label" toName="image">\n'
        f'    <Label value="{label_name}" background="#FF0000"/>\n'
        f"  </RectangleLabels>\n"
        f"</View>"
    )


def validate_label_studio_config(
    xml_content: str,
    expected_label: str = DEFAULT_LABEL_NAME,
) -> bool:
    """Validate that XML string is well-formed and has required Label Studio tags.

    Args:
        xml_content: The XML configuration string.
        expected_label: Expected label value to find in RectangleLabels.

    Returns:
        True if configuration is valid.

    Raises:
        ValueError: If required tags or labels are missing or malformed.
    """
    try:
        root = ET.fromstring(xml_content.strip())
    except ET.ParseError as e:
        raise ValueError(f"Malformed XML configuration: {e}") from e

    if root.tag != "View":
        raise ValueError(f"Root element must be <View>, got <{root.tag}>")

    image_tag = root.find("Image")
    if image_tag is None:
        raise ValueError("Missing <Image> tag in configuration")
    if image_tag.get("name") != "image":
        raise ValueError(f"Expected <Image name=\"image\">, got {image_tag.attrib}")

    rect_labels = root.find("RectangleLabels")
    if rect_labels is None:
        raise ValueError("Missing <RectangleLabels> tag in configuration")
    if rect_labels.get("toName") != "image":
        raise ValueError("Expected <RectangleLabels toName=\"image\">")

    labels = rect_labels.findall("Label")
    label_values = [lbl.get("value") for lbl in labels]
    if expected_label not in label_values:
        raise ValueError(
            f"Expected label '{expected_label}' not found among {label_values}"
        )

    return True
