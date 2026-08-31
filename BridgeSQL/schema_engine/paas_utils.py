import datetime
import decimal


def examples_to_str(examples: list) -> list[str]:
    """Convert schema examples to printable strings, excluding URLs."""
    values = examples
    for i in range(len(values)):
        if isinstance(values[i], datetime.date):
            values = [values[i]]
            break
        elif isinstance(values[i], decimal.Decimal):
            values[i] = str(float(values[i]))
        elif "http://" in str(values[i]) or "https://" in str(values[i]):
            values = []
            break

    return [str(v) for v in values if v is not None and len(str(v)) > 0]
