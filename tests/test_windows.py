"""Structural regression tests; no model weights, private data or GPU needed."""
import unittest
from prepare_data import transactions,plan_windows
from export_messages import from_atif


class WindowTests(unittest.TestCase):
    def test_complete_parallel_tool_transaction(self):
        messages=[{'role':'user','content':'task'},
            {'role':'assistant','tool_calls':[{'id':'a','function':{'name':'shell'}},{'id':'b','function':{'name':'read'}}]},
            {'role':'tool','tool_call_id':'b','name':'read','content':'ok'},
            {'role':'tool','tool_call_id':'a','name':'shell','content':'ok'},
            {'role':'assistant','content':'done'}]
        self.assertEqual(transactions(messages),(1,[(1,4),(4,5)]))
        for broken in [messages[:3],messages[:2]+[messages[2],messages[2],messages[4]]]:
            with self.assertRaises(AssertionError):transactions(broken)

    def test_withholding_does_not_cut_or_repeat_targets(self):
        lengths=[4,4,20,3,4,2]
        plans,withheld=plan_windows(lengths,2,12)
        targets=[i for _,start,end in plans for i in range(start,end)]
        omitted=[i for i,_ in withheld]
        self.assertEqual(sorted(targets+omitted),list(range(len(lengths))))
        self.assertEqual(len(set(targets)),len(targets))
        self.assertIn((2,'complete_transaction_exceeds_budget'),withheld)
        self.assertIn((3,'immediate_complete_history_does_not_fit'),withheld)
        for history,target,end in plans:
            self.assertLessEqual(sum(lengths[history:end])+2,12)
            self.assertTrue(target==0 or history<target)

    def test_exact_original_observation_only(self):
        trace={'steps':[{'source':'user','message':'task'},
            {'source':'agent','message':'','reasoning_content':'plan',
             'tool_calls':[{'tool_call_id':'c','function_name':'shell','arguments':{'command':'echo 2'}}]}]}
        request={'tools':[{'type':'function','function':{'name':'shell'}}],
            'messages':[{'role':'system','content':'instructions'},
                        {'role':'tool','tool_call_id':'c','content':'original output'}]}
        messages,_,restored=from_atif(trace,request)
        self.assertEqual(restored,1);self.assertEqual(messages[-1]['content'],'original output')
        self.assertEqual(messages[-2]['reasoning_content'],'plan')
        request['messages']=request['messages'][:1]
        with self.assertRaises(ValueError):from_atif(trace,request)


if __name__=='__main__':unittest.main()
