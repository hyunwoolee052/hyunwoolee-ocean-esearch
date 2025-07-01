def validate_date(date_str: str) -> bool:
    from datetime import datetime
    
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False

def validate_param_name(param_name: str, valid_params: list) -> bool:
    return param_name in valid_params

def log_message(message: str) -> None:
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logging.info(message)