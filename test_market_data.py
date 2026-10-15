from datetime import datetime
from zoneinfo import ZoneInfo
from market_data import call_rows, iso_time, tradier_snapshot


def test_call_rows_keep_standard_calls():
    q = [
      dict(option_type='call',underlying='HIMZ',root_symbol='HIMZ',contract_size=100,
           strike=25,bid=1.10,ask=1.24,open_interest=500,volume=4,symbol='HIMZ261120C00025000',
           bid_date=1762982138000,greeks={'smv_vol':.63}),
      dict(option_type='call',underlying='HIMZ',root_symbol='HIMZ1',contract_size=10,
           strike=25,bid=8,ask=9,symbol='HIMZ1261120C00025000'),
      dict(option_type='put',underlying='HIMZ',root_symbol='HIMZ',contract_size=100,
           strike=24,bid=.5,ask=.7,symbol='HIMZ261120P00024000'),
      dict(option_type='call',underlying='HIMZ',root_symbol='HIMZ',contract_size=100,
           strike=26,bid=0,ask=1.5,symbol='HIMZ261120C00026000'),
    ]
    rows=call_rows('2026-11-20',q,'HIMZ')
    assert len(rows)==1
    assert rows[0]['bid']==1.1
    assert rows[0]['iv']==.63
    assert rows[0]['bid_time'].startswith('2025-')


def test_iso_time_handles_milliseconds():
    assert iso_time(1762982138000)==iso_time(1762982138)
    assert iso_time(0) is None


def test_expiry_retrieval_without_network(monkeypatch):
    import market_data as m
    dt=datetime.now(ZoneInfo('America/New_York')).date()
    expiration=(dt+__import__('datetime').timedelta(days=28)).isoformat()
    paths=[]
    def fake_req(tok,endpoint,params,sandbox=False):
        paths.append(endpoint)
        if endpoint.endswith('/quotes'):
            return {'quotes':{'quote':{'last':8.15,'ask':8.20,'bid':8.10,'ask_date':1762982138000}}}
        if endpoint.endswith('/expirations'):
            return {'expirations':{'date':[expiration]}}
        if endpoint.endswith('/history'):
            return {'history':{'day':[]}}
        if endpoint.endswith('/chains'):
            return {'options':{'option':[dict(option_type='call',root_symbol='BB',underlying='BB',contract_size=100,
                                strike=9,bid=.60,ask=.70,open_interest=160,volume=30,symbol='BB261120C00009000',
                                greeks={'smv_vol':.50})]}}
        raise AssertionError(endpoint)
    monkeypatch.setattr(m,'tradier_get',fake_req)
    res=m.tradier_snapshot('BB',21,45,'example')
    assert res['spot']==8.2
    assert res['rows'][0]['strike']==9
    assert len(paths)==4


def test_probability_requires_valid_data():
    from trade_math import assess
    r=assess(8.2,9.0,.60,.70,28,.5,1000,160)
    assert abs(r['cost']-820)<1e-9
    assert abs(r['max_profit']-140)<1e-9
    assert 0 < r['p_profit'] < 1
