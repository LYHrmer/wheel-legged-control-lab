"""Actual XTest events on the launcher's private X server, no policy/physics."""
from __future__ import annotations

import ctypes as C
import json
import os
from pathlib import Path
import time


class EventDriver:
    def __init__(self, output, spec):
        self.output = Path(output)
        self.spec = spec
        self.events = []
        self.index = 0
        self.display = None
        self.helper = self.window = 0
        self.held = set()
        name = os.environ.get('DISPLAY','')
        if not name.startswith(':') or not 100 <= int(name[1:].split('.')[0]) < 200:
            raise RuntimeError('XTest is restricted to the owned :100..:199 server')

    def initialize(self, ready):
        self.x = C.CDLL('libX11.so.6')
        self.xt = C.CDLL('libXtst.so.6')
        pointer = C.c_void_p
        window = C.c_ulong
        signatures = {
            'XOpenDisplay':([C.c_char_p],pointer),
            'XDefaultRootWindow':([pointer],window),
            'XCreateSimpleWindow':([pointer,window,C.c_int,C.c_int,C.c_uint,C.c_uint,C.c_uint,window,window],window),
            'XMapWindow':([pointer,window],C.c_int),
            'XSetInputFocus':([pointer,window,C.c_int,window],C.c_int),
            'XStringToKeysym':([C.c_char_p],window),
            'XKeysymToKeycode':([pointer,window],C.c_uint),
            'XSync':([pointer,C.c_int],C.c_int),
            'XDestroyWindow':([pointer,window],C.c_int),
            'XCloseDisplay':([pointer],C.c_int),
        }
        for name,(args,result) in signatures.items():
            getattr(self.x,name).argtypes=args
            getattr(self.x,name).restype=result
        self.xt.XTestFakeKeyEvent.argtypes=[pointer,C.c_uint,C.c_int,window]
        self.xt.XTestFakeKeyEvent.restype=C.c_int
        self.display=self.x.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError('cannot open private X11 display')
        self.window=int(ready['x11_window'])
        self.buttons=ready.get('buttons',{})
        self.x.XWarpPointer.argtypes=[pointer,window,window,C.c_int,C.c_int,C.c_uint,C.c_uint,C.c_int,C.c_int]
        self.x.XWarpPointer.restype=C.c_int
        self.xt.XTestFakeButtonEvent.argtypes=[pointer,C.c_uint,C.c_int,window]
        self.xt.XTestFakeButtonEvent.restype=C.c_int
        self.helper=self.x.XCreateSimpleWindow(self.display,self.x.XDefaultRootWindow(self.display),
                                              1100,100,160,120,0,0,0)
        self.x.XMapWindow(self.display,self.helper)
        self.x.XSetInputFocus(self.display,self.window,1,0)
        self.x.XSync(self.display,0)
        self.events.append({'operation':'initial_focus','wall_ns':time.monotonic_ns(),'window':self.window})
        with (self.output/'event_driver_ready.json').open('x') as stream:
            json.dump({'ready':True,'private_display':os.environ['DISPLAY'],
                       'wall_ns':time.monotonic_ns()},stream)

    def key(self,name,down):
        code=self.x.XKeysymToKeycode(self.display,self.x.XStringToKeysym(name.encode()))
        if not code or not self.xt.XTestFakeKeyEvent(self.display,code,int(down),0):
            raise RuntimeError('XTest key injection failed: '+name)
        if down:
            self.held.add(name)
        else:
            self.held.discard(name)

    def advance(self):
        ready_path=self.output/'window_ready.json'
        if self.display is None:
            if not ready_path.exists():
                return
            try:
                ready=json.loads(ready_path.read_text())
            except json.JSONDecodeError:
                return
            self.initialize(ready)
        progress_path=self.output/'live_status.json'
        if not progress_path.exists():
            return
        try:
            progress=json.loads(progress_path.read_text())
        except json.JSONDecodeError:
            return
        completed=int(progress['completed_controls'])
        while self.index<len(self.spec) and completed>=self.spec[self.index]['tick']:
            row=self.spec[self.index]
            if row['operation']=='focus_lost':
                self.x.XSetInputFocus(self.display,self.helper,1,0)
            elif row['operation']=='focus_return':
                self.x.XSetInputFocus(self.display,self.window,1,0)
            elif row['operation'] in ('key_down','key_up'):
                self.key(row['key'],row['operation']=='key_down')
            elif row['operation']=='button_click':
                target=self.buttons[row['button']]
                x,y=int(target['x']),int(target['y'])
                self.x.XWarpPointer(self.display,0,self.window,0,0,0,0,x,y)
                self.x.XSync(self.display,0)
                if not self.xt.XTestFakeButtonEvent(self.display,1,1,0):
                    raise RuntimeError('XTest mouse press failed')
                if not self.xt.XTestFakeButtonEvent(self.display,1,0,0):
                    raise RuntimeError('XTest mouse release failed')
            else:
                raise ValueError('unknown X11 operation')
            self.x.XSync(self.display,0)
            self.events.append({**row,'observed_completed_controls':completed,
                                'wall_ns':time.monotonic_ns()})
            self.index+=1

    def close(self):
        if self.display is not None:
            for name in tuple(self.held):
                self.key(name,False)
            if self.helper:
                self.x.XDestroyWindow(self.display,self.helper)
            self.x.XSync(self.display,0)
            self.x.XCloseDisplay(self.display)
            self.display=None
        with (self.output/'x11_events_receipt.json').open('x') as stream:
            json.dump({'schema':'d1-c23-real-xtest-events-v1','sent_count':self.index,
                       'expected_count':len(self.spec),'events':self.events,
                       'all_sent':self.index==len(self.spec),
                       'physical_human_keyboard_tested':False},stream,indent=2)
            stream.write('\n')
