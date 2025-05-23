import logging

def loggingassert(condition, message, errtyp=ValueError):
    '''
    a simple wrapper for assert with logging
    '''
    if not condition:
        logging.error(message)
        raise errtyp(message)
