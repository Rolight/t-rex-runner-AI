import numpy as np
import time
import json

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.options import Options


class INVALID_ACTION(Exception):
    pass


class GameEnv:
    # Driver is supposed to running in offine mode.
    # You can use this url to run controller in online mode.
    URL = 'http://wayou.github.io/t-rex-runner/'

    def __init__(self, offline=True, headless=False):
        if headless:
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument("--window-size=1920x1080")
            self.driver = webdriver.Chrome(chrome_options=chrome_options)
            offline = False
        else:
            self.driver = webdriver.Chrome()
        # make driver offine
        if offline:
            self.driver.set_network_conditions(
                offline=True, latency=5, throughput=500 * 1024)
        self.actions = ActionChains(self.driver)
        self.keys = Keys
        self._init_game()
        '''
        action space:
        [action_id, hold_time]
        action_id: 0 - no action, 1 - jump, 2 - squant
        '''
        self.action_dim = 3
        self.action_id_space = [0, 1, 2]
        self.max_hold_time = 1
        '''
            observation space: [
                obstacle type
                xpos of nearest obstacle,
                ypos of nearest obstacle,
                width of nearest obstacle,
                current speed,
            ]
        '''
        self.observation_dim = 5
        self.FPS = 60
        self.interval_time = 1 / self.FPS

        self.HOLDING_KEYCODE = {
            'DOWN': 40,
            'UP': 38,
        }

        self.HOLDING_KEY_NAME = {
            'DOWN': 'ArrowDown',
            'UP': 'ArrowUP',
        }

    def _init_game(self):
        # open the game
        self.driver.get(self.URL)
        self.game_element = self.driver.find_element_by_class_name(
            'runner-canvas')

    def __del__(self):
        self.driver.quit()

    def _driver_dispatchKeyEvent(self, name, options={}):
        if self.status == 'CRASHED':
            return
        driver = self.driver
        options["type"] = name
        body = json.dumps({'cmd': 'Input.dispatchKeyEvent', 'params': options})
        resource = "/session/%s/chromium/send_command" % driver.session_id
        url = driver.command_executor._url + resource
        driver.command_executor._request('POST', url, body)

    # Since Chrome Driver doesn't support hold a key,
    # We need to write a function to make driver hold key press
    # option's format is from follow link:
    # https://chromedevtools.github.io/devtools-protocol/tot/Input/
    def _driver_holdKey(self, duration, code, key, text, keycode):
        # if duration is None, we assume it is a speeddrop process
        if duration is not None:
            endtime = time.time() + duration
        options = {
            "code": code,
            "key": key,
            "text": text,
            "unmodifiedText": text,
            "nativeVirtualKeyCode": keycode,
            "windowsVirtualKeyCode": keycode,
            "autoRepeat": True
        }

        self._driver_dispatchKeyEvent("rawKeyDown", options)
        while True:
            # self._driver_dispatchKeyEvent("char", options)

            if duration is not None and (time.time() > endtime or self.status == 'CRASHED'):
                self._driver_dispatchKeyEvent("keyUp", options)
                break

            if duration is None and (self.status == 'CRASHED' or self.status == 'DUCKING'):
                self._driver_dispatchKeyEvent("keyUp", options)
                break

            options["autoRepeat"] = True
            time.sleep(self.interval_time)

    def _send_key_to_game(self, key, holding=False, holding_duration=None):
        if not holding:
            self.actions.key_down(key, element=self.game_element).perform()
            self.actions.reset_actions()
        else:
            if key not in self.HOLDING_KEYCODE:
                print('invalid hold key, only support holding UP and DOWN')
                return
            self.actions.click(self.game_element).perform()
            self.actions.reset_actions()
            self._driver_holdKey(
                duration=holding_duration,
                code=self.HOLDING_KEY_NAME[key],
                key=self.HOLDING_KEY_NAME[key],
                text="",
                keycode=self.HOLDING_KEYCODE[key]
            )

    def _send_keyup_to_game(self, key):
        self.actions.key_up(key, element=self.game_element).perform()
        self.actions.reset_actions()

    def _get_runner_state(self, state):
        return self.driver.execute_script('return Runner.instance_.%s' % state)

    def _exec_runner_func(self, func):
        return self.driver.execute_script('return Runner.instance_.%s()' % func)

    @property
    def activated(self):
        return self._get_runner_state('activated')

    @property
    def isRunning(self):
        return self._exec_runner_func('isRunning')

    @property
    def isJumping(self):
        return self._get_runner_state('tRex.jumping')

    @property
    def status(self):
        return self._get_runner_state('tRex.status')

    def get_observation(self):
        obstacles = self._get_runner_state('horizon.obstacles')
        if len(obstacles) == 0:
            return None
        nearest_obstacle = obstacles[0]
        obstacle_type, x_pos, y_pos, width = (
            1 if nearest_obstacle['typeConfig']['type'] == 'PTERODACTYL' else 0,
            nearest_obstacle['xPos'],
            nearest_obstacle['yPos'],
            nearest_obstacle['width'])
        cur_speed = self._get_runner_state('currentSpeed')
        return np.array([obstacle_type, x_pos, y_pos, width, cur_speed])

    def start(self):
        self._send_key_to_game(self.keys.SPACE)

    def restart(self):
        self._exec_runner_func('restart')

    def perform_action(self, action, verbose=False):
        if verbose:
            print('perform action ', action)
        if self.status != 'RUNNING':
            return None, None, None
        action_id = action[0]
        hold_time = action[1]
        if action_id not in self.action_id_space:
            raise INVALID_ACTION
        hold_time = min(hold_time, self.max_hold_time)
        if action_id == 0:
            # nothing
            time.sleep(hold_time)
        elif action_id == 1:
            # jump
            self._send_key_to_game(self.keys.UP)
            time.sleep(hold_time)
            self._send_key_to_game('DOWN', holding=True, holding_duration=None)
        elif action_id == 2:
            # squant
            self._send_key_to_game('DOWN', holding=True,
                                   holding_duration=hold_time)
        obs = self.get_observation()
        reward = 1
        done = False
        if self.status == 'CRASHED':
            reward = -1
            done = True
        return obs, done, reward

    def sample_action(self):
        action_id = np.random.randint(0, self.action_dim)
        hold_time = np.random.uniform(0, self.max_hold_time)
        return np.array([action_id, hold_time])
