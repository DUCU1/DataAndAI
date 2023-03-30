from durable.lang import *
with ruleset('expense'):
    @when_any(all(c.first << m.subject == 'approve', c.second << m.amount == 1000),
              all(c.third << m.subject == 'jumbo', c.fourth << m.amount == 10000))

    def action(c):
        if c.first:
         print ('Approved {0} {1}'.format(c.first.subject, c.second.amount))
        else:
         print ('Approved {0} {1}'.format(c.third.subject, c.fourth.amount))

post('expense', { 'subject': 'approve' })
post('expense', { 'amount': 1000 })
post('expense', { 'subject': 'jumbo' })
post('expense', { 'amount': 10000 })