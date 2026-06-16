
def sec2ddhhmm(t: int) -> str:
    """
    Convert seconds to 'DDdHHhMMm' format (zero-padded).
    Example: 100000 -> '01d03h46m'
    """
    t = int(t)
    days = t // (24 * 3600)
    hours = (t % (24 * 3600)) // 3600
    minutes = (t % 3600) // 60

    return f"{days:d}d{hours:02d}h{minutes:02d}m"
