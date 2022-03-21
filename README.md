## About The Code

> To Do : Write Code Documentation

## Versioning

> To Do: Delete later

### When To Branch:

Check the status to make sure everything is clean.

`$ git status`

`On branch master`<br />
`nothing to commit, working tree clean`

If we use the -b flag, then we will create a new branch and also switch to that branch:

`$ git checkout -b make_function`

`Switched to a new branch 'make_function'`

after staging and committing changes on the new branch, comes the merge.

we can bring the contents of the `make_function` branch into the `master` branch by using `git merge`

but first we we need to check out the branch we want to marge into (i.e `master`)

`$ git checkout master`

`Switched to branch 'master'`

then

`$ git merge make_function`

`Updating 3f62d8f..1071b15`<br />
`Fast-forward`<br />
`word_count.py | 23 ++++++++++++++++-------`<br />
`1 file changed, 16 insertions(+), 7 deletions(-)`<br />

Finally we can delete the branch by new branch

`$ git branch -d make_function `

`Deleted branch make_function (was 1071b15).`
