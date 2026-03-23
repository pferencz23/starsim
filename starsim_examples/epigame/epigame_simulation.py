import starsim as ss
import matplotlib.pyplot as plt
import numpy as np
import sciris as sc

ss_int = ss.dtypes.int
ss_float = ss.dtypes.float
_ = None # For function signatures

class SEIR(ss.Infection):
    """
    SEIR model

    This class implements a basic SEIR model with states for susceptible, exposed,
    infected/infectious, and recovered. It also includes deaths, and basic
    results.

    Args:
        beta (float/`ss.prob`): the infectiousness
        init_prev (float/s`s.bernoulli`): the fraction of people to start of being infected
        dur_exp (float/`ss.dur`/`ss.Dist`): how long (in years) people are exposed/incubating for
        dur_inf (float/`ss.dur`/`ss.Dist`): how long (in years) people are infected for
        p_death (float/`ss.bernoulli`): the probability of death from infection
    """
    def __init__(self, pars=None, beta=_, init_prev=_, dur_inf=_, p_death=_, **kwargs):
        super().__init__()
        self.define_pars(
            beta = ss.peryear(0.1),
            init_prev = ss.bernoulli(p=0.01),
            dur_exp = ss.lognorm_ex(mean=ss.years(1)),
            dur_inf = ss.lognorm_ex(mean=ss.years(6)),
            p_death = ss.bernoulli(p=0.01),
        )
        self.update_pars(pars, **kwargs)

        # Example of defining all states, redefining those from ss.Infection, using overwrite=True
        self.define_states(
            ss.BoolState('susceptible', default=True, label='Susceptible'),
            ss.BoolState('exposed', label='Exposed'),
            ss.BoolState('infected', label='Infectious'),
            ss.BoolState('recovered', label='Recovered'),
            ss.FloatArr('ti_exposed', label='TIme of exposure'),
            ss.FloatArr('ti_infected', label='Time of infection'),
            ss.FloatArr('ti_recovered', label='Time of recovery'),
            ss.FloatArr('ti_dead', label='Time of death'),
            ss.FloatArr('rel_sus', default=1.0, label='Relative susceptibility'),
            ss.FloatArr('rel_trans', default=1.0, label='Relative transmission'),
            reset = True, # Remove any existing states (from super().define_states())
        )
        return

    def step_state(self):
        # Progress exposed -> infected
        sim = self.sim
        infected = self.exposed & (self.ti_infected <= sim.ti)
        self.exposed[infected] = False
        self.infected[infected] = True

        # Progress infectious -> recovered
        recovered = (self.infected & (self.ti_recovered <= sim.ti)).uids
        self.infected[recovered] = False
        self.recovered[recovered] = True

        # Trigger deaths
        deaths = (self.ti_dead <= sim.ti).uids
        if len(deaths):
            sim.people.request_death(deaths)
        return

    def set_prognoses(self, uids, sources=None):
        """ Set prognoses """
        super().set_prognoses(uids, sources)
        ti = self.t.ti
        self.susceptible[uids] = False
        self.exposed[uids] = True
        self.infected[uids] = False
        self.ti_exposed[uids] = ti

        p = self.pars

        # Sample duration of exposed/incubation
        dur_exp = p.dur_exp.rvs(uids)
        self.ti_infected[uids] = ti + dur_exp

        # Sample duration of infection
        dur_inf = p.dur_inf.rvs(uids)

        # Determine who dies and who recovers and when
        will_die = p.p_death.rvs(uids)
        dead_uids = uids[will_die]
        rec_uids = uids[~will_die]
        self.ti_dead[dead_uids] = ti + dur_exp[will_die] + dur_inf[will_die] # Consider rand round, but not CRN safe
        self.ti_recovered[rec_uids] = ti + dur_exp[~will_die] + dur_inf[~will_die]
        return

    def step_die(self, uids):
        """ Reset infected/recovered flags for dead agents """
        self.susceptible[uids] = False
        self.exposed[uids] = False
        self.infected[uids] = False
        self.recovered[uids] = False
        return

    def plot(self, **kwargs):
        """ Default plot for SIR model """
        fig = plt.figure()
        kw = sc.mergedicts(dict(lw=2, alpha=0.8), kwargs)
        res = self.results
        for rkey in ['n_susceptible', 'n_exposed', 'n_infected', 'n_recovered']:
            plt.plot(res.timevec, res[rkey], label=res[rkey].label, **kw)
        plt.legend(frameon=False)
        plt.xlabel('Time')
        plt.ylabel('Number of people')
        plt.ylim(bottom=0)
        sc.boxoff()
        sc.commaticks()
        return ss.return_fig(fig)


#### Define simulation parameters
# TODO: update based on epigame parameters
people = ss.People(n_agents=1000)
network = ss.RandomNet(n_contacts=2) # TODO: replace with AUIB data-based network
seir = SEIR(
        init_prev = ss.bernoulli(p=0.01),
        beta = ss.perday(0.0907*24),
        dur_inf = ss.lognorm_ex(mean=ss.days(77/24), std=ss.days(0.5)),
        p_death = ss.bernoulli(p=0.6*0.25 + 0.4*0.7),
        dur_exp = ss.lognorm_ex(mean=ss.days(10/24), std=ss.days(0.2)),
    )

# Run simulation
sim = ss.Sim(
    start='2020-01-01', 
    stop='2020-01-15', 
    dt=ss.days(1),          
    diseases=seir, 
    networks=network,
    people=people
    )

sim.run()
sim.plot()
sim.diseases.seir.plot()