.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/sibirrer/lightcone/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

lightcone could always use more documentation, whether as part of the
official lightcone docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/sibirrer/lightcone/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

GitHub Workflow
---------------

Fork and Clone the lightcone Repository
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
**You should only need to do this step once**

First *fork* the lightcone repository. A fork is your own remote copy of the repository on GitHub. To create a fork:

  1. Go to the `lightcone Repository <https://github.com/sibirrer/lightcone>`_
  2. Click the **Fork** button (in the top-right-hand corner)
  3. Choose where to create the fork, typically your personal GitHub account

Next *clone* your fork. Cloning creates a local copy of the repository on your computer to work with. To clone your fork:

::

   git clone https://github.com/<your-account>/lightcone.git


Finally add the ``lightcone-project`` repository as a *remote*. This will allow you to fetch changes made to the codebase.
To add the ``lightcone-project`` remote:

::

  cd lightcone
  git remote add lightcone-project https://github.com/sibirrer/lightcone.git


Create a branch for your new feature
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a *branch* off the ``lightcone-project`` main branch.
Working on unique branches for each new feature simplifies the development, review and merge processes by maintaining logical separation.
To create a feature branch:

::

  git fetch lightcone-project
  git checkout -b <your-branch-name> lightcone-project/main


Hack away!
^^^^^^^^^^

Write the new code you would like to contribute and *commit* it to the feature branch on your local repository.
Ideally commit small units of work often with clear and descriptive commit messages describing the changes you made. To commit changes to a file:

::

  git add file_containing_your_contribution
  git commit -m 'Your clear and descriptive commit message'


*Push* the contributions in your feature branch to your remote fork on GitHub:

::

  git push origin <your-branch-name>


**Note:** The first time you *push* a feature branch you will probably need to use `--set-upstream origin` to link to your remote fork:

::

  git push --set-upstream origin <your-branch-name>


Open a Pull Request
^^^^^^^^^^^^^^^^^^^

When you feel that work on your new feature is complete, you should create a *Pull Request*. This will propose your work to be merged into the main sim-pipeline repository.

  1. Go to `lightcone Pull Requests <https://github.com/sibirrer/lightcone/pulls>`_
  2. Click the green **New pull request** button
  3. Click **compare across forks**
  4. Confirm that the base fork is ``sibirrer/lightcone`` and the base branch is ``main``
  5. Confirm the head fork is ``<your-account>/lightcone`` and the compare branch is ``<your-branch-name>``
  6. Give your pull request a title and fill out the the template for the description
  7. Click the green **Create pull request** button

Status checks
^^^^^^^^^^^^^

A series of automated checks will be run on your pull request, some of which will be required to pass before it can be merged into the main codebase:

  - ``Tests`` (Required) runs the `unit tests`_ in four predefined environments; `latest supported versions`, `oldest supported versions`, `macOS latest supported` and `Windows latest supported`. Click "Details" to view the output including any failures.
  - ``Code Style`` (Required) runs `flake8 <https://flake8.pycqa.org/en/latest/>`__ to check that your code conforms to the `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ style guidelines. Click "Details" to view any errors.
  - ``codecov`` reports the test coverage for your pull request; you should aim for `codecov/patch — 100.00%`. Click "Details" to view coverage data.
  - ``docs`` (Required) builds the `docstrings`_ on `readthedocs <https://readthedocs.org/>`_. Click "Details" to view the documentation or the failed build log.

Updating your branch
^^^^^^^^^^^^^^^^^^^^

As you work on your feature, new commits might be made to the ``lightcone-project`` main branch.
You will need to update your branch with these new commits before your pull request can be accepted.
You can achieve this in a few different ways:

  - If your pull request has no conflicts, click **Update branch**
  - If your pull request has conflicts, click **Resolve conflicts**, manually resolve the conflicts and click **Mark as resolved**
  - *merge* the ``lightcone-project`` main branch from the command line:

    ::

        git fetch lightcone-project
        git merge lightcone-project/main

  - *rebase* your feature branch onto the ``lightcone-project`` main branch from the command line:
    ::

        git fetch lightcone-project
        git rebase lightcone-project/main


**Warning**: It is bad practice to *rebase* commits that have already been pushed to a remote such as your fork.
Rebasing creates new copies of your commits that can cause the local and remote branches to diverge. ``git push --force`` will **overwrite** the remote branch with your newly rebased local branch.
This is strongly discouraged, particularly when working on a shared branch where you could erase a collaborators commits.

For more information about resolving conflicts see the GitHub guides:
  - `Resolving a merge conflict on GitHub <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/resolving-a-merge-conflict-on-github>`_
  - `Resolving a merge conflict using the command line <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/resolving-a-merge-conflict-using-the-command-line>`_
  - `About Git rebase <https://help.github.com/en/github/using-git/about-git-rebase>`_

More Information
^^^^^^^^^^^^^^^^

More information regarding the usage of GitHub can be found in the `GitHub Guides <https://guides.github.com/>`_.

Coding Guidelines
-----------------

Before your pull request can be merged into the codebase, it will be reviewed by one of the sim-pipeline developers and required to pass a number of automated checks. Below are a minimum set of guidelines for developers to follow:

General Guidelines
^^^^^^^^^^^^^^^^^^

- lightcone is compatible with Python>=3.9 (see `setup.cfg <https://github.com/sibirrer/lightcone/blob/main/setup.cfg>`_). sim-pipeline *does not* support backwards compatibility with Python 2.x; `six`, `__future__` and `2to3` should not be used.
- All contributions should follow the `PEP8 Style Guide for Python Code <https://www.python.org/dev/peps/pep-0008/>`_. We recommend using `flake8 <https://flake8.pycqa.org/>`__ to check your code for PEP8 compliance.
- Importing lightcone should only depend on having `NumPy <https://www.numpy.org>`_, `SciPy <https://www.scipy.org/>`_ and `Astropy <https://www.astropy.org/>`__ installed.
- Code will be grouped into submodules based on broad applications.
- For more information see the `Astropy Coding Guidelines <http://docs.astropy.org/en/latest/development/codeguide.html>`_.

Unit Tests
^^^^^^^^^^

Pull requests will require existing unit tests to pass before they can be merged.
Additionally, new unit tests should be written for all new public methods and functions.
Unit tests for each submodule are contained in subdirectories called ``tests`` and you can run them locally using ``pytest``.
For more information see the `Astropy Testing Guidelines <https://docs.astropy.org/en/stable/development/testguide.html>`_.

If your unit tests check the statistical distribution of a random sample, the test outcome itself is a random variable, and the test will fail from time to time.
Please mark such tests with the ``@pytest.mark.flaky`` decorator, so that they will be automatically tried again on failure. To prevent non-random test failures from being run multiple times, please isolate random statistical tests and deterministic tests in their own test cases.

Docstrings
^^^^^^^^^^

All public classes, methods and functions require docstrings. You can build documentation locally by installing `sphinx-astropy <https://github.com/astropy/sphinx-astropy>`_ and calling ``make html`` in the ``docs`` subdirectory.
Docstrings should include the following sections:

  - Description
  - Parameters
  - Notes
  - References

For more information see the Astropy guide to `Writing Documentation <https://docs.astropy.org/en/stable/development/docguide.html>`_.
