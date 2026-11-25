import asyncio
import binascii
import time
from typing import Any
from unittest.mock import Mock, call

import pytest

from nibe.coil import Coil, CoilData
from nibe.connection.nibegw import ConnectionStatus, NibeGW, xor8
from nibe.exceptions import (
    NoMappingException,
    ReadException,
    ReadTimeoutException,
    WriteException,
)
from nibe.heatpump import HeatPump, Model, ProductInfo


@pytest.fixture(name="heatpump")
async def fixture_heatpump() -> HeatPump:
    heatpump = HeatPump(Model.F1255)
    heatpump.word_swap = True
    await heatpump.initialize()
    return heatpump


@pytest.fixture(name="transport")
async def fixture_transport():
    return Mock()


@pytest.fixture(name="nibegw")
async def fixture_nibegw(heatpump: HeatPump, transport: Mock):
    nibegw = NibeGW(heatpump, "127.0.0.1")
    assert nibegw.status == ConnectionStatus.UNKNOWN
    nibegw.connection_made(transport)
    return nibegw


def _enqueue_datagram(nibegw: NibeGW, data: str):
    asyncio.get_event_loop().call_soon(
        nibegw.datagram_received, binascii.unhexlify(data), ("127.0.0.1", 12345)
    )


def _modbus_response(command: int, data: bytes) -> bytes:
    escaped = data.replace(b"\x5c", b"\x5c\x5c")
    body = bytes((0, 0x20, command, len(escaped))) + escaped
    return b"\x5c" + body + bytes((xor8(body),))


def _receive_logset(nibegw: NibeGW, words: list[tuple[int, int]]) -> None:
    payload = b"".join(
        address.to_bytes(2, "little") + word.to_bytes(2, "little")
        for address, word in words
    )
    nibegw.datagram_received(_modbus_response(0x68, payload), ("127.0.0.1", 12345))


@pytest.mark.parametrize("word_swap", [False, True, None])
async def test_logset_word_order(
    nibegw: NibeGW, heatpump: HeatPump, word_swap: bool | None
):
    nibegw.coil_encoder.word_swap = word_swap
    coil = heatpump.get_coil_by_address(43424)
    received = Mock()
    heatpump.subscribe(HeatPump.COIL_UPDATE_EVENT, received)

    _receive_logset(nibegw, [(coil.address, 0x1234), (coil.address + 1, 3)])

    received.assert_called_once_with(CoilData(coil, 201268))


@pytest.mark.parametrize("upper_slots", ["present", "missing", "duplicate_low"])
async def test_phase_current_logset_recovery(
    nibegw: NibeGW, heatpump: HeatPump, upper_slots: str
):
    nibegw.coil_encoder.word_swap = False
    coils = [heatpump.get_coil_by_address(address) for address in (40079, 40081, 40083)]
    received = Mock()
    heatpump.subscribe(HeatPump.COIL_UPDATE_EVENT, received)

    words = []
    for coil in coils:
        words.append((coil.address, 123))
        if upper_slots == "present":
            words.append((coil.address + 1, 0))
        else:
            words.append((0xFFFF, 0))
        if upper_slots == "duplicate_low":
            words.append((coil.address, 124))
            words.append((0xFFFF, 0))

    _receive_logset(nibegw, words)

    if upper_slots == "duplicate_low":
        received.assert_not_called()
    else:
        assert received.call_args_list == [call(CoilData(coil, 12.3)) for coil in coils]


@pytest.mark.parametrize("size", ["u16", "u32"])
async def test_unknown_register_continuation_skipped(
    heatpump: HeatPump, monkeypatch: pytest.MonkeyPatch, size: str
):
    coil = Coil(
        32477,
        "energy",
        "Energy",
        size,
        low_word_sufficient=(size == "u32"),
    )
    next_coil = Coil(32478, "hw-energy", "HW Energy", "u16")
    monkeypatch.setitem(heatpump._address_to_coil, str(coil.address), coil)
    monkeypatch.setitem(heatpump._address_to_coil, str(next_coil.address), next_coil)
    nibegw = NibeGW(heatpump)
    received = Mock()
    heatpump.subscribe(HeatPump.COIL_UPDATE_EVENT, received)

    # 32476 is unknown, so 32477 (and 32478 when 32477 is u32) may be continuations.
    _receive_logset(nibegw, [(32476, 5), (32477, 123), (32478, 1)])
    if size == "u32":
        received.assert_not_called()
    else:
        received.assert_called_once_with(CoilData(next_coil, 1))

    received.reset_mock()
    _receive_logset(nibegw, [(32476, 5), (0xFFFF, 0), (32477, 123)])
    received.assert_called_once_with(CoilData(coil, 123))


@pytest.mark.parametrize("layout", ["separated", "reversed"])
async def test_logset_requires_consecutive_slots_for_complete_pair(
    nibegw: NibeGW, heatpump: HeatPump, layout: str
):
    coil = heatpump.get_coil_by_address(43424)
    good_coil = heatpump.get_coil_by_address(40004)
    received = Mock()
    heatpump.subscribe(HeatPump.COIL_UPDATE_EVENT, received)
    words = [(coil.address, 123), (good_coil.address, 150), (coil.address + 1, 1)]
    if layout == "reversed":
        words.reverse()

    _receive_logset(nibegw, words)

    received.assert_called_once_with(CoilData(good_coil, 15))


@pytest.mark.parametrize(
    ("collision", "neighbor_words", "mode", "reason"),
    [
        (False, [], "permissive", "incomplete"),
        (False, [0, 251], "permissive", "ambiguous"),
        (True, [251], "permissive", "ambiguous"),
        (True, [251], "strict", "ambiguous"),
        (True, [], "strict", "incomplete"),
    ],
)
async def test_ignore_incomplete_or_ambiguous_broadcast_pairs(
    nibegw: NibeGW,
    heatpump: HeatPump,
    caplog: pytest.LogCaptureFixture,
    collision: bool,
    neighbor_words: list[int],
    mode: str,
    reason: str,
):
    coil = heatpump.get_coil_by_address(43424)
    coil.f_series_word_collision = collision
    good_coil = heatpump.get_coil_by_address(40004)
    received = Mock()
    heatpump.subscribe(HeatPump.COIL_UPDATE_EVENT, received)
    nibegw._table_processing_mode = mode
    words = [(coil.address, 1500)]
    words.extend((coil.address + 1, word) for word in neighbor_words)
    words.append((good_coil.address, 123))
    with caplog.at_level("DEBUG"):
        _receive_logset(nibegw, words)

    if mode == "permissive":
        received.assert_called_once_with(CoilData(good_coil, 12.3))
    else:
        received.assert_not_called()
    assert (
        f"Skipping coil {coil.address} - {reason} 32-bit broadcast" in caplog.messages
    )


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        # Only first register in LOG.SET (post-reboot slot 2 carries neighbor's word)
        ([(32473, 100), (32474, 200)], [(32473, 0.1)]),
        # Both adjacent u32 registers in LOG.SET before reboot
        (
            [(32473, 100), (32474, 0), (32474, 200), (32475, 0)],
            [(32473, 0.1), (32474, 0.2)],
        ),
        # Both adjacent u32 registers in LOG.SET after reboot
        (
            [(32473, 100), (32474, 200), (32474, 200), (32475, 50)],
            [(32473, 0.1), (32474, 0.2)],
        ),
    ],
)
async def test_logset_adjacent_colliding_u32_registers(
    heatpump: HeatPump,
    monkeypatch: pytest.MonkeyPatch,
    words: list[tuple[int, int]],
    expected: list[tuple[int, float]],
):
    heatpump.word_swap = False
    nibegw = NibeGW(heatpump, table_processing_mode="strict")
    coils = {
        address: Coil(
            address,
            f"energy-{address}",
            f"Energy {address}",
            "u32",
            factor=1000,
            f_series_word_collision=True,
            low_word_sufficient=True,
        )
        for address in (32473, 32474)
    }
    for address, coil in coils.items():
        monkeypatch.setitem(heatpump._address_to_coil, str(address), coil)
    received = Mock()
    heatpump.subscribe(HeatPump.COIL_UPDATE_EVENT, received)

    _receive_logset(nibegw, words)

    assert received.call_args_list == [
        call(CoilData(coils[address], value)) for address, value in expected
    ]


@pytest.mark.parametrize(("low_word", "expected"), [(0x1234, 4.66), (0xFFFF, None)])
async def test_read_only_the_low_word(
    nibegw: NibeGW, heatpump: HeatPump, low_word: int, expected: float | None
):
    coil = heatpump.get_coil_by_address(43424)
    coil.size = "u32"
    coil.factor = 1000
    coil.f_series_word_collision = True
    coil.low_word_sufficient = True
    payload = (
        coil.address.to_bytes(2, "little")
        + low_word.to_bytes(2, "little")
        + b"\xfb\x00"
    )
    _enqueue_datagram(nibegw, _modbus_response(0x6A, payload).hex())

    assert await nibegw.read_coil(coil, timeout=0.1) == CoilData(coil, expected)


@pytest.mark.parametrize(
    ("low_word_sufficient", "word_swap", "message"),
    [
        (True, False, "word swap"),
        (True, None, "word swap"),
        (False, True, "second word"),
    ],
)
async def test_reject_colliding_direct_read(
    nibegw: NibeGW,
    heatpump: HeatPump,
    transport: Mock,
    low_word_sufficient: bool,
    word_swap: bool | None,
    message: str,
):
    coil = heatpump.get_coil_by_address(43424)
    coil.f_series_word_collision = True
    coil.low_word_sufficient = low_word_sufficient
    nibegw.coil_encoder.word_swap = word_swap

    with pytest.raises(ReadException, match=message):
        await nibegw.read_coil(coil)

    transport.sendto.assert_not_called()


async def test_read_coil_with_incomplete_broadcast(nibegw: NibeGW, heatpump: HeatPump):
    coil = heatpump.get_coil_by_address(43416)
    _enqueue_datagram(
        nibegw,
        "5c00206850449c38004f9c8a00509c5f004e9c7c014d9ca301489c10014c9cfe00ada93c0080a8c80098a9ad0cffff0000ec9f52ffffff00000aad41a1ffff000012ad2399ffff0000889c870001a810014ea81e0002",
    )
    _enqueue_datagram(nibegw, "5c00206a0698a9ad0c0000dc")

    data = await nibegw.read_coil(coil)

    assert data.value == 3245


async def test_status(nibegw: NibeGW, heatpump: HeatPump):
    assert nibegw.status == ConnectionStatus.LISTENING

    connection_status_handler_mock = Mock()
    nibegw.subscribe(NibeGW.CONNECTION_STATUS_EVENT, connection_status_handler_mock)

    coil = heatpump.get_coil_by_address(43424)

    _enqueue_datagram(nibegw, "5c00206a06a0a9f5120000a2")

    await nibegw.read_coil(coil)

    assert nibegw.status == ConnectionStatus.CONNECTED
    connection_status_handler_mock.assert_called_once_with(
        status=ConnectionStatus.CONNECTED
    )

    connection_status_handler_mock.reset_mock()
    _enqueue_datagram(nibegw, "5c00206a06a0a9f5120000a2")
    await nibegw.read_coil(coil)
    connection_status_handler_mock.assert_not_called()


async def test_read_s32_coil(nibegw: NibeGW, heatpump: HeatPump, transport: Mock):
    coil = heatpump.get_coil_by_address(43424)

    _enqueue_datagram(nibegw, "5c00206a06a0a9f5120000a2")

    coil_data = await nibegw.read_coil(coil)
    assert coil_data.value == 4853

    transport.sendto.assert_called_once_with(
        binascii.unhexlify("c06902a0a9a2"), ("127.0.0.1", 9999)
    )


async def test_write_s32_coil(nibegw: NibeGW, heatpump: HeatPump, transport: Mock):
    coil = heatpump.get_coil_by_address(40940)
    coil_data = CoilData(coil, -10)

    _enqueue_datagram(nibegw, "5c00206c01014c")
    await nibegw.write_coil(coil_data)

    transport.sendto.assert_called_once_with(
        binascii.unhexlify("c06b06ec9f9cffffffbd"), ("127.0.0.1", 10000)
    )


async def test_read_coil_decode_failed(
    nibegw: NibeGW, heatpump: HeatPump, transport: Mock
):
    coil = heatpump.get_coil_by_address(43086)

    _enqueue_datagram(nibegw, "5c00206a064ea8f51200004d")

    start = time.time()
    data = await nibegw.read_coil(coil, timeout=0.1)
    assert data.value == "UNKNOWN (245)"

    with pytest.raises(NoMappingException):
        data.validate()

    assert 1 == transport.sendto.call_count
    duration = time.time() - start
    assert duration <= 0.1


async def test_read_coil_timeout(nibegw: NibeGW, heatpump: HeatPump, transport: Mock):
    coil = heatpump.get_coil_by_address(43086)

    start = time.time()
    with pytest.raises(ReadTimeoutException):
        await nibegw.read_coil(coil, timeout=0.1)
    duration = time.time() - start
    assert (
        0.3 <= duration <= 0.4
    ), "Timeout should be between 0.3 and 0.4 seconds. We do 3 retries"
    assert 3 == transport.sendto.call_count, "Should do 3 retries"
    transport.sendto.assert_called_with(b"\xc0i\x02N\xa8M", ("127.0.0.1", 9999))


async def test_write_coil(nibegw: NibeGW, heatpump: HeatPump, transport: Mock):
    coil = heatpump.get_coil_by_address(48132)
    coil_data = CoilData(coil, "One time increase")

    _enqueue_datagram(nibegw, "5c00206c01014c")
    await nibegw.write_coil(coil_data)

    transport.sendto.assert_called_once_with(
        binascii.unhexlify("c06b0604bc0400000011"), ("127.0.0.1", 10000)
    )


async def test_write_coil_failed(nibegw: NibeGW, heatpump: HeatPump, transport: Mock):
    coil = heatpump.get_coil_by_address(48132)
    coil_data = CoilData(coil, "One time increase")

    _enqueue_datagram(nibegw, "5c00206c01004d")
    with pytest.raises(WriteException):
        await nibegw.write_coil(coil_data)

    assert 1 == transport.sendto.call_count, "Should only send once, no retry"


async def test_read_product_info(nibegw: NibeGW):
    _enqueue_datagram(nibegw, "5c00206d0d0124e346313235352d313220529f")
    product = await nibegw.read_product_info()

    assert isinstance(product, ProductInfo)
    assert "F1255-12 R" == product.model


async def test_read_product_info_with_extras(nibegw: NibeGW):
    _enqueue_datagram(
        nibegw,
        "5c0019ee00f7"  # token accessory version
        "c0ee03ee0101c3"  # accessory version from accessory
        "06"  # ack from pump
        "5c00206d0d0124e346313235352d313220529f",
    )
    product = await nibegw.read_product_info()

    assert isinstance(product, ProductInfo)
    assert "F1255-12 R" == product.model


@pytest.mark.parametrize(
    ("raw", "table_processing_mode", "calls"),
    [
        (
            "5c00206850449c38004f9c8a00509c5f004e9c7c014d9ca301489c10014c9cfe00ada93c0080a8c80098a9ad0cffff0000ec9f52ffffff00000aad41a1ffff000012ad2399ffff0000889c870001a810014ea81e0002",
            "permissive",
            [
                (40004, 5.6),
                (40008, 27.2),
                (40012, 25.4),
                (40013, 41.9),
                (40014, 38.0),
                (40015, 13.8),
                (40016, 9.5),
                (40072, 13.5),
                (43009, 27.2),
                (43086, "HEAT"),
                (43136, 20.0),
                (43437, 60),
            ],
        ),
        (
            "5c00206850449c38004f9c8a00509c5f004e9c7c014d9ca301489c10014c9cfe00ada93c0080a8c80098a9ad0cffff0000ec9f52ffffff00000aad41a1ffff000012ad2399ffff0000889c870001a810014ea81e0002",
            "strict",
            [],
        ),
        (
            "5c00206850c9af0000889c7100a9a90a00a3a91400aba90000939c0000949c0000919c3c00929c00008f9c0000909c00003ab95000ada94600a7a91400faa90200ffff0000ffff0000ffff0000ffff0000ffff0000f0",
            "strict",
            [
                (40072, 11.3),
                (40079, 0.0),
                (40081, 6.0),
                (40083, 0.0),
                (43427, "STOPPED"),
                (43431, "ON"),
                (43433, "OFF"),
                (43435, "OFF"),
                (43437, 70),
                (43514, 2),
                (45001, 0),
                (47418, 80),
            ],
        ),
        (
            "5c00206850c9af0000889c7100a9a90a00a3a91400aba90000939c0000949c0000919c3c00929c00008f9c0000909c0000 3ab97000 ada94600 a7a91400 faa90200 ffff0000 ffff0000 ffff0000 ffff0000 ffff0000 d0",
            "permissive",
            [
                (40072, 11.3),
                (40079, 0.0),
                (40081, 6.0),
                (40083, 0.0),
                (43427, "STOPPED"),
                (43431, "ON"),
                (43433, "OFF"),
                (43435, "OFF"),
                (43437, 70),
                (43514, 2),
                (45001, 0),
                # (47418, 80), # This coil will be ignored because of decoding error
            ],
        ),
        (
            "5c00206850c9af0000889c7100a9a90a00a3a91400aba90000939c0000949c0000919c3c00929c00008f9c0000909c0000 3ab97000 ada94600 a7a91400 faa90200 ffff0000 ffff0000 ffff0000 ffff0000 ffff0000 d0",
            "strict",
            [
                # All will be ignored because of single decoding error
            ],
        ),
        (
            "5c00206850 489ce400 4c9ce300 4e9ca101 889c4500 d5a1ae00 d6a1a300 fda718f8 c5a5ad98c6a50100 cda5d897cea50100 cfa51fb7d0a50600 98a96d23 99a90000 a0a9cf05 a1a90000 9ca9a01a 9da90000 449c4500 e5",
            "strict",
            [
                (40004, 6.9),
                (40008, 22.8),
                (40012, 22.7),
                (40014, 41.7),
                (40072, 6.9),
                (41429, 17.4),
                (41430, 16.3),
                (
                    42437,
                    10462.1,
                ),  # 32-bit register occupies two addresses (42437, 42438): c5a5 ad98 c6a5 0100
                (
                    42445,
                    10440.8,
                ),  # 32-bit register occupies two addresses (42445, 42446): cda5 d897 cea5 0100
                (
                    42447,
                    44009.5,
                ),  # 32-bit register occupies two addresses (42447, 42448): cfa5 1fb7 d0a5 0600
                (43005, -202.4),
                (43416, 9069),
                (43420, 6816),
                (43424, 1487),
            ],
        ),
    ],
)
async def test_read_multiple_with_u32(
    nibegw: NibeGW,
    heatpump: HeatPump,
    raw: str,
    table_processing_mode: str,
    calls: list[tuple[int, Any]],
):
    nibegw._table_processing_mode = table_processing_mode
    on_coil_update_mock = Mock()
    heatpump.subscribe("coil_update", on_coil_update_mock)
    nibegw.datagram_received(
        binascii.unhexlify(raw.replace(" ", "")),
        ("127.0.0.1", 12345),
    )

    def _call(address, value):
        return call(CoilData(heatpump.get_coil_by_address(address), value))

    # Values with 0000 will not be included in the call list
    assert on_coil_update_mock.mock_calls == [_call(*call) for call in calls]
