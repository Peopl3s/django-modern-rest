from http import HTTPStatus

import pytest
from django.conf import LazySettings

from dmr.errors import ErrorDetail, ErrorType, format_error
from dmr.exceptions import (
    InternalServerError,
    NotAcceptableError,
    NotAuthenticatedError,
    RequestSerializationError,
    ResponseSchemaError,
    ValidationError,
)


def test_format_error_from_string() -> None:
    """Ensures plain string is wrapped into a detail entry."""
    formatted = format_error('something went wrong')
    assert formatted == {'detail': [{'msg': 'something went wrong'}]}


def test_format_error_from_string_with_loc() -> None:
    """Ensures ``loc`` parameter is included as a list in the detail entry."""
    formatted = format_error('bad value', loc='body')
    assert formatted == {'detail': [{'msg': 'bad value', 'loc': ['body']}]}


def test_format_error_with_error_type() -> None:
    """Ensures ``error_type`` parameter is included in the detail entry."""
    formatted = format_error('bad value', error_type=ErrorType.user_msg)
    assert formatted == {'detail': [{'msg': 'bad value', 'type': 'user_msg'}]}


def test_format_error_with_loc_and_type() -> None:
    """Ensures both ``loc`` and ``error_type`` are included together."""
    formatted = format_error(
        'bad value',
        loc='field',
        error_type=ErrorType.value_error,
    )
    assert formatted == {
        'detail': [
            {'msg': 'bad value', 'loc': ['field'], 'type': 'value_error'},
        ],
    }


def test_format_error_error_type_as_str() -> None:
    """Ensures ``error_type`` accepts plain strings."""
    formatted = format_error('oops', error_type='custom_type')
    assert formatted == {'detail': [{'msg': 'oops', 'type': 'custom_type'}]}


def test_format_error_from_validation() -> None:
    """Ensures ``ValidationError`` payload is returned as-is."""
    payload: list[ErrorDetail] = [
        {'msg': 'too short', 'loc': ['name'], 'type': 'value_error'},
    ]
    exc = ValidationError(payload, status_code=HTTPStatus.UNPROCESSABLE_ENTITY)

    formatted = format_error(exc)

    assert formatted == {'detail': payload}


def test_format_error_from_request_serial() -> None:
    """Ensures ``RequestSerializationError`` gets ``value_error`` type."""
    exc = RequestSerializationError('cannot parse body')

    formatted = format_error(exc)

    assert formatted == {
        'detail': [{'msg': 'cannot parse body', 'type': 'value_error'}],
    }


def test_format_error_from_response_schema() -> None:
    """Ensures ``ResponseSchemaError`` gets ``value_error`` type."""
    exc = ResponseSchemaError('schema mismatch')

    formatted = format_error(exc)

    assert formatted == {
        'detail': [{'msg': 'schema mismatch', 'type': 'value_error'}],
    }


def test_format_error_from_not_acceptable() -> None:
    """Ensures ``NotAcceptableError`` gets ``value_error`` type."""
    exc = NotAcceptableError('unsupported media type')

    formatted = format_error(exc)

    assert formatted == {
        'detail': [{'msg': 'unsupported media type', 'type': 'value_error'}],
    }


def test_format_error_from_not_authed() -> None:
    """Ensures ``NotAuthenticatedError`` gets ``security`` type."""
    exc = NotAuthenticatedError('token expired')

    formatted = format_error(exc)

    assert formatted == {
        'detail': [{'msg': 'token expired', 'type': 'security'}],
    }


def test_format_error_not_authed_default() -> None:
    """Ensures ``NotAuthenticatedError`` uses default message."""
    exc = NotAuthenticatedError()

    formatted = format_error(exc)

    assert formatted == {
        'detail': [{'msg': 'Not authenticated', 'type': 'security'}],
    }


def test_format_error_from_ise_debug(
    settings: LazySettings,
) -> None:
    """Ensures ``InternalServerError`` shows details in DEBUG mode."""
    settings.DEBUG = True
    exc = InternalServerError('database is down')

    formatted = format_error(exc)

    assert formatted == {'detail': [{'msg': 'database is down'}]}


def test_format_error_from_ise_no_debug(
    settings: LazySettings,
) -> None:
    """Ensures ``InternalServerError`` hides details without DEBUG."""
    settings.DEBUG = False
    exc = InternalServerError('database is down')

    formatted = format_error(exc)

    assert formatted == {
        'detail': [{'msg': InternalServerError.default_message}],
    }


def test_format_error_unknown_exc_raises() -> None:
    """Ensures unhandled exception types raise ``NotImplementedError``."""
    with pytest.raises(NotImplementedError, match='Cannot format error'):
        format_error(ValueError('unexpected'))
