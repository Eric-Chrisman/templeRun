from pyautogui import *
import pyautogui
import time
import keyboard
import numpy as np
import win32api, win32con

# controls for a bot to play temple run with
# is not the bot

leaningLeft = False
leaningRight = False

# clicks somewhere
def click(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
    time.sleep(0.1)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)

# Function to handle leaning left
# if leaning right, stop
# true = lean left 
# false = go center
def leanLeft(tempBool: bool = True):
    global leaningLeft, leaningRight
    if leaningRight:
        keyboard.release('x')
    if tempBool:
        keyboard.press('z')
        leaningLeft = True
    else:
        keyboard.release('z')
        leaningLeft = False

# Function to handle leaning right
# if leaning left, stop
# true = lean right
# false = go center
def leanRight(tempBool: bool = True):
    global leaningLeft, leaningRight
    if leaningLeft:
        keyboard.release('z')
    if tempBool:
        keyboard.press('x')
        leaningRight = True
    else:
        keyboard.release('x')
        leaningRight = False

# jump
def swipeUp():
    keyboard.press("up")
    sleep(0.01)
    keyboard.release("up")

# slide
def swipeDown():
    keyboard.press("down")
    sleep(0.01)
    keyboard.release("down")

# trun left
def swipeLeft():
    keyboard.press("left")
    sleep(0.01)
    keyboard.release("left")

# turn right
def swipeRight():
    keyboard.press("right")
    sleep(0.01)
    keyboard.release("right")

# check if dead. If dead, reset
def tryReset() -> bool:
    try:
        pyautogui.locateOnScreen("youDied.png", grayscale=True, confidence=0.8)
        print("dead")
        return True
    except pyautogui.ImageNotFoundException:
        print("not dead")
    return False

# assumes that you are on score screen
def resetMacro():
    click(150, 550)
    sleep(2)
    click(260,800)

# assumes that you are on main menu. should olny be call at very very start of program
def startMacro():
    click(200,500)
    click(200,500)

# don't call; displays mouse data forever. olny for mapping screen
def displayMousePosition():
    pyautogui.displayMousePosition()

def perform_action(action):
    if action == 'jump':
        swipeUp()
    elif action == 'slide':
        swipeDown()
    elif action == 'go_left':
        swipeLeft()
    elif action == 'go_right':
        swipeRight()
    elif action == 'lean_left':
        leanLeft()
    elif action == 'lean_center':
        leanLeft(False)
        leanRight(False)
    elif action == 'lean_right':
        leanRight()