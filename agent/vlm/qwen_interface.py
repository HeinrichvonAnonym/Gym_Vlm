
import openai
from time import sleep
# from openai.error import RateLimitError, APIConnectionError
from openai import RateLimitError, APIConnectionError
from openai import AsyncOpenAI, OpenAI
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter
from agent.vlm.utils import load_prompt, DynamicObservation, IterableDynamicObservation
import time
from agent.vlm.LLM_cache import DiskCache

class LMP:
    """Language Model Program (LMP), adopted from Code as Policies."""
    def __init__(self, name, cfg, fixed_vars, variable_vars, debug=False, env=''):
        self._name = name
        self._cfg = cfg
        self._debug = debug
        self._base_prompt = load_prompt(f"{env}/{self._cfg['prompt_fname']}.txt")
        self._stop_tokens = list(self._cfg['stop'])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''
        self._context = None
        self._cache = DiskCache(load_cache=self._cfg['load_cache'])

        # 初始化 QWEN 客户端
        self.client = OpenAI(
            api_key="sk-615a37ab09c04416b45567f99d10de2e",  # 替换为你的 QWEN API Key
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # QWEN 兼容端点
        )

    def clear_exec_hist(self):
        self.exec_hist = ''

    def build_prompt(self, query):
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = f"from utils import {', '.join(self._variable_vars.keys())}"
        else:
            variable_vars_imports_str = ''
        prompt = self._base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)

        if self._cfg['maintain_session'] and self.exec_hist != '':
            prompt += f'\n{self.exec_hist}'
        
        prompt += '\n'  # separate prompted examples with the query part

        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'
            prompt += f'\n{self._context}'

        user_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f'\n{user_query}'

        return prompt, user_query
    
    def _cached_api_call(self, **kwargs):
        # 替换模型名称为 QWEN 模型
        qwen_models = {
            'gpt-4': 'qwen-max',
            'gpt-3.5-turbo': 'qwen-plus',
            'gpt-3.5-turbo-instruct': 'qwen-turbo'
        }
        
        original_model = kwargs.get('model', '')
        # 将 GPT 模型映射到 QWEN 模型
        for gpt_model, qwen_model in qwen_models.items():
            if gpt_model in original_model:
                kwargs['model'] = qwen_model
                break
        
        # 检查是否使用聊天端点
        if any([model in kwargs['model'] for model in ['qwen-max', 'qwen-plus', 'qwen-turbo']]):
            # 构建消息格式(与原代码相同)
            user1 = kwargs.pop('prompt')
            new_query = '# Query:' + user1.split('# Query:')[-1]
            user1 = ''.join(user1.split('# Query:')[:-1]).strip()
            user1 = f"我希望你帮我编写 Python 代码来控制在桌面环境中操作的机械臂。每次我给你新查询时,请完成代码。注意给定上下文代码中出现的模式。在你的代码中要全面和深思熟虑。不要包含任何导入语句。不要重复我的问题。不要提供任何文本解释(代码中的注释可以)。我首先给你下面的代码上下文:\n\n```\n{user1}\n```\n\n注意 x 是从后到前,y 是从左到右,z 是从下到上。"
            
            assistant1 = '明白了。我会完成你接下来给我的内容。'
            user2 = new_query
            
            # 处理对象上下文
            if user1.split('\n')[-4].startswith('objects = ['):
                obj_context = user1.split('\n')[-4]
                user1 = '\n'.join(user1.split('\n')[:-4]) + '\n' + '\n'.join(user1.split('\n')[-3:])
                user2 = obj_context.strip() + '\n' + user2
            
            messages = [
                {"role": "system", "content": "你是一个有帮助的助手,会注意用户的指令并为在桌面环境中操作机械臂编写良好的 Python 代码。"},
                {"role": "user", "content": user1},
                {"role": "assistant", "content": assistant1},
                {"role": "user", "content": user2},
            ]
            
            # 检查缓存
            cache_key = (kwargs['model'], str(messages), kwargs.get('temperature'), kwargs.get('max_tokens'))
            if cache_key in self._cache:
                print('(使用缓存)', end=' ')
                return self._cache[cache_key]
            
            # 调用 QWEN API
            response = self.client.chat.completions.create(
                model=kwargs['model'],
                messages=messages,
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 512),
                stop=kwargs.get('stop', None)
            )
            
            ret = response.choices[0].message.content
            ret = ret.replace('```', '').replace('python', '').strip()
            self._cache[cache_key] = ret
            return ret
        else:
            # 处理补全端点(如果需要)
            cache_key = (kwargs['model'], kwargs.get('prompt'), kwargs.get('temperature'), kwargs.get('max_tokens'))
            if cache_key in self._cache:
                print('(使用缓存)', end=' ')
                return self._cache[cache_key]
            
            response = self.client.completions.create(
                model=kwargs['model'],
                prompt=kwargs['prompt'],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 512),
                stop=kwargs.get('stop', None)
            )
            
            ret = response.choices[0].text.strip()
            self._cache[cache_key] = ret
            return ret

    def __call__(self, query, **kwargs):
        prompt, user_query = self.build_prompt(query)

        start_time = time.time()
        while True:
            try:
                code_str = self._cached_api_call(
                    prompt=prompt,
                    stop=self._stop_tokens,
                    temperature=self._cfg['temperature'],
                    model=self._cfg['model'],
                    max_tokens=self._cfg['max_tokens']
                )
                break
            except (RateLimitError, APIConnectionError) as e:
                print(f'OpenAI API got err {e}')
                print('Retrying after 3s.')
                sleep(3)
        print(f'*** OpenAI API call took {time.time() - start_time:.2f}s ***')

        if self._cfg['include_context']:
            assert self._context is not None, 'context is None'
            to_exec = f'{self._context}\n{code_str}'
            to_log = f'{self._context}\n{user_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{user_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())

        if self._cfg['include_context']:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + f'## context: "{self._context}"\n' + '#'*40 + f'\n{to_log_pretty}\n')
        else:
            print('#'*40 + f'\n## "{self._name}" generated code\n' + '#'*40 + f'\n{to_log_pretty}\n')

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        # return function instead of executing it so we can replan using latest obs（do not do this for high-level UIs)
        if not self._name in ['composer', 'planner']:
            to_exec = 'def ret_val():\n' + to_exec.replace('ret_val = ', 'return ')
            to_exec = to_exec.replace('\n', '\n    ')

        if self._debug:
            # only "execute" function performs actions in environment, so we comment it out
            action_str = ['execute(']
            try:
                for s in action_str:
                    exec_safe(to_exec.replace(s, f'# {s}'), gvars, lvars)
            except Exception as e:
                print(f'Error: {e}')
                import pdb ; pdb.set_trace()
        else:
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f'\n{to_log.strip()}'

        if self._cfg['maintain_session']:
            self._variable_vars.update(lvars)

        if self._cfg['has_return']:
            if self._name == 'parse_query_obj':
                try:
                    # there may be multiple objects returned, but we also want them to be unevaluated functions so that we can access latest obs
                    return IterableDynamicObservation(lvars[self._cfg['return_val_name']])
                except AssertionError:
                    return DynamicObservation(lvars[self._cfg['return_val_name']])
            return lvars[self._cfg['return_val_name']]


def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }
    

def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str
  
    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        print(f'Error executing code:\n{code_str}')
        raise e
    

if __name__ == '__main__':
    print("="*50)
    print("LMP + QWEN API 测试程序")
    print("="*50)
    
    # 配置参数
    cfg = {
      "prompt_fname": "planner_prompt",
      "model": "gpt-4",
      "max_tokens": 512,
      "temperature": 0,
      "query_prefix": '# Query: ',
      "query_suffix": '.',
      "stop":
         ['# Query: ','objects = '],
      "maintain_session": False,
      "include_context": True,
      "has_return": False,
      "return_val_name": "ret_val",
      "load_cache": True,
    }
    
    # 初始化 LMP
    print("\n[1] 初始化 LMP 实例...")
    lmp = LMP(
        name='test_lmp',
        cfg=cfg,
        fixed_vars={},
        variable_vars={},
        debug=False
    )
    print("✓ LMP 初始化成功!\n")
    
    